#
# Restarted from the original DeepDPM ClusterNetModel (Meitar Ronen, March 2022).
# Copyright (c) 2022 Meitar Ronen
#
# This version keeps the ORIGINAL split/merge/GMM machinery completely untouched
# (that logic was confirmed, through debugging, to be independent of the
# contrastive addition below — a contrastive_weight=0 baseline plateaus at the
# same K as contrastive_weight=1, so the split/merge dynamics are not where the
# pairwise-supervision concept should hook in).
#
# On top of the original, this adds ONE clean extension: optional pairwise
# (same/different) supervision, active only when the dataloader yields
# (x, y, z) triples instead of (x, y) pairs. Design choices made deliberately
# to avoid bugs found in earlier iterations:
#
#   1. mus/covs/pi are ALWAYS recomputed on schedule, regardless of whether
#      contrastive supervision is active. There is no "contrastive_only" mode
#      that permanently freezes the GMM fit — that was the single biggest bug
#      found in earlier versions (K got stuck at 1 for an entire 300-epoch run
#      because comp_cluster_params was gated behind `not contrastive_only`
#      with no time limit).
#   2. The contrastive loss is computed on softmax(logits), which DOES have a
#      gradient path back into cluster_net — not on raw codes, which (in the
#      embeddings-only / no-feature_extractor setting) are leaf tensors with
#      no gradient path and would make the pairwise term a no-op that only
#      logs a number.
#   3. contrastive_weight is an explicit, separate knob from cluster_loss_weight
#      (previously the pairwise term's effective weight was implicitly
#      (1 - cluster_loss_weight), which silently zeroed it out under the
#      library's own default of cluster_loss_weight=1).
#   4. The optional subcluster-level pairwise term only compares pairs whose
#      anchor and positive/negative currently share the same PREDICTED
#      top-level cluster. Comparing subcluster responsibilities across two
#      different predicted clusters is a trivial/degenerate comparison (the
#      two one-hot-ish vectors have disjoint support and are always maximally
#      far apart), so it contributes no real gradient signal and was pruned.
#
# Everything else — init, gather_codes, split_step, merge_step, priors,
# optimizers, plotting, logging — is the untouched original.

from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
from torch import optim
import pytorch_lightning as pl
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score, silhouette_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure

from src.clustering_models.clusternet_modules.utils.plotting_utils import PlotUtils
from src.clustering_models.clusternet_modules.utils.training_utils import training_utils
from src.clustering_models.clusternet_modules.utils.clustering_utils.priors import (
    Priors,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.clustering_operations import (
    init_mus_and_covs,
    compute_data_covs_hard_assignment,
)
from src.clustering_models.clusternet_modules.utils.clustering_utils.split_merge_operations import (
    update_models_parameters_split,
    split_step,
    merge_step,
    update_models_parameters_merge,
)
from src.clustering_models.clusternet_modules.models.Classifiers import MLP_Classifier, Subclustering_net


class ClusterNetModel(pl.LightningModule):
    def __init__(self, hparams, input_dim, init_k, feature_extractor=None, n_sub=2, centers=None, init_num=0):
        """The main class of the unsupervised clustering scheme.
        Performs all the training steps.

        Args:
            hparams ([namespace]): model-specific hyperparameters
            input_dim (int): the shape of the input data
            train_dl (DataLoader): The dataloader to train on
            init_k (int): The initial K to start the net with
            feature_extractor (nn.Module): The feature extractor to get codes with
            n_sub (int, optional): Number of subclusters per cluster. Defaults to 2.
        """

        super().__init__()
        self.hparams = hparams
        self.K = init_k
        self.n_sub = n_sub
        self.codes_dim = input_dim
        self.split_performed = False  # indicator to know whether a split was performed
        self.merge_performed = False
        self.feature_extractor = feature_extractor
        self.centers = centers
        if self.hparams.seed:
            pl.utilities.seed.seed_everything(self.hparams.seed)

        # initialize cluster_net
        self.cluster_net = MLP_Classifier(hparams, k=self.K, codes_dim=self.codes_dim)

        if not self.hparams.ignore_subclusters:
            # initialize subclustering net
            self.subclustering_net = Subclustering_net(hparams, codes_dim=self.codes_dim, k=self.K)
        else:
            self.subclustering_net = None
        self.last_key = self.K - 1  # variable to help with indexing the dict

        self.training_utils = training_utils(hparams)
        self.last_val_NMI = 0
        self.init_num = init_num
        self.prior_sigma_scale = self.hparams.prior_sigma_scale
        if self.init_num > 0 and self.hparams.prior_sigma_scale_step != 0:
            self.prior_sigma_scale = self.hparams.prior_sigma_scale / (self.init_num * self.hparams.prior_sigma_scale_step)
        self.use_priors = self.hparams.use_priors
        self.prior = Priors(hparams, K=self.K, codes_dim=self.codes_dim, prior_sigma_scale=self.prior_sigma_scale)  # we will use for split and merges even if use_priors is false

        self.mus_inds_to_merge = None
        self.mus_ind_to_split = None

        # ---- training history for post-training diagnostics plot ----
        self.history = {
            "epoch": [],
            "K": [],
            "cluster_loss": [],
            "subcluster_loss": [],
            "pairwise_loss": [],
            "sub_pairwise_loss": [],
        }
        self._epoch_cluster_losses = []
        self._epoch_subcluster_losses = []
        self._epoch_pairwise_losses = []
        self._epoch_sub_pairwise_losses = []

    # ------------------------------------------------------------------
    # Pairwise contrastive loss (new — the pairwise-supervision concept)
    # ------------------------------------------------------------------
    def contrastive_loss(self, z1, z2, pair_label, margin=1.0):
        """Contrastive loss over a pair of vectors (typically softmax cluster
        assignments). pair_label: 1 = same cluster/class, 0 = different.

        Args:
            z1, z2 (Tensor): [N, D] paired vectors (e.g. softmax(logits))
            pair_label (Tensor): [N] float, 1 for positive pairs, 0 for negative
            margin (float): margin for negative pairs
        """
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        dist = F.pairwise_distance(z1, z2)

        positive_loss = pair_label.float() * torch.pow(dist, 2)
        negative_loss = (1 - pair_label.float()) * torch.pow(torch.clamp(margin - dist, min=0.0), 2)

        return torch.mean(positive_loss + negative_loss)

    def forward(self, x):
        codes_a, _ = self._extract_codes(x)
        return self.cluster_net(codes_a)

    def on_train_epoch_start(self):
        # reset per-epoch loss trackers
        self._epoch_cluster_losses = []
        self._epoch_subcluster_losses = []
        self._epoch_pairwise_losses = []
        self._epoch_sub_pairwise_losses = []
        # get current training_stage
        self.current_training_stage = (
            "gather_codes" if self.current_epoch == 0 and not hasattr(self, "mus") else "train_cluster_net"
        )
        self.initialize_net_params(stage="train")
        if self.split_performed or self.merge_performed:
            self.split_performed = False
            self.merge_performed = False

    def on_validation_epoch_start(self):
        self.initialize_net_params(stage="val")
        return super().on_validation_epoch_start()

    def initialize_net_params(self, stage="train"):
        self.codes = []
        if stage == "train":
            if self.current_epoch > 0:
                del self.train_resp, self.train_resp_sub, self.train_gt
            self.train_resp = []
            self.train_resp_sub = []
            self.train_gt = []
        else:
            if self.current_epoch > 0:
                del self.val_resp, self.val_resp_sub, self.val_gt
            self.val_resp = []
            self.val_resp_sub = []
            self.val_gt = []

    # ------------------------------------------------------------------
    # Helper: unpack a batch into (codes_a, codes_b). codes_b is None
    # unless the batch is a pair (x has an extra "pair" dimension) AND a
    # feature_extractor isn't being used to project on the fly in a way
    # that only handles a single view (kept in sync with the original
    # single-view behaviour when there is no pairing).
    # ------------------------------------------------------------------
    def _extract_codes(self, x):
        if self.feature_extractor is not None:
            with torch.no_grad():
                if x.ndim == 3:
                    x_a = x[:, 0, :]
                    codes_a = torch.from_numpy(
                        self.feature_extractor(x_a.view(x_a.size()[0], -1), latent=True)
                    ).to(device=self.device)
                    x_b = x[:, 1, :]
                    codes_b = torch.from_numpy(
                        self.feature_extractor(x_b.view(x_b.size()[0], -1), latent=True)
                    ).to(device=self.device)
                else:
                    codes_a = torch.from_numpy(
                        self.feature_extractor(x.view(x.size()[0], -1), latent=True)
                    ).to(device=self.device)
                    codes_b = None
        elif x.ndim == 3:
            codes_a = x[:, 0, :]
            codes_b = x[:, 1, :]
        else:
            codes_a = x
            codes_b = None
        return codes_a, codes_b

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if batch[0].ndim == 3:
            x, y, z = batch
        else:
            x, y = batch
            z = None

        codes_a, codes_b = self._extract_codes(x)

        if self.current_training_stage == "gather_codes":
            return self.only_gather_codes(codes_a, y, optimizer_idx)
        elif self.current_training_stage == "train_cluster_net":
            return self.cluster_net_pretraining(codes_a, codes_b, y, z, optimizer_idx, x if batch_idx == 0 else None)
        else:
            raise NotImplementedError()

    def only_gather_codes(self, codes, y, optimizer_idx):
        """Only log codes for initialization"""
        if optimizer_idx == self.optimizers_dict_idx["cluster_net_opt"]:
            (
                self.codes,
                self.train_gt,
                _,
                _,
            ) = self.training_utils.log_codes_and_responses(
                model_codes=self.codes,
                model_gt=self.train_gt,
                model_resp=self.train_resp,
                model_resp_sub=self.train_resp_sub,
                codes=codes,
                y=y,
                logits=None,
            )
        return None

    def cluster_net_pretraining(self, codes, codes_b, y, z, optimizer_idx, x_for_vis=None):
        """Pretraining function for the clustering and subclustering nets.
        The GMM fit (cluster_loss) is always active. If paired data with
        pair labels z is provided, an auxiliary pairwise contrastive term is
        added on top with its own explicit weight — it never replaces or
        gates the GMM fit.
        """
        codes = codes.reshape(-1, self.codes_dim)

        logits = self.cluster_net(codes)
        cluster_loss = self.training_utils.cluster_loss_function(
            codes,
            logits,
            model_mus=self.mus,
            K=self.K,
            codes_dim=self.codes_dim,
            model_covs=self.covs if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
            pi=self.pi,
            logger=self.logger,
        )
        self.log("cluster_net_train/train/cluster_number", self.K)
        self.log(
            "cluster_net_train/train/cluster_loss",
            self.hparams.cluster_loss_weight * cluster_loss,
            on_epoch=True,
        )
        self._epoch_cluster_losses.append(cluster_loss.detach().item())
        loss = self.hparams.cluster_loss_weight * cluster_loss

        # ---- optional pairwise contrastive term (top-level clusters) ----
        if codes_b is not None and z is not None and self.hparams.contrastive_weight > 0:
            codes_b = codes_b.reshape(-1, self.codes_dim)
            logits_b = self.cluster_net(codes_b)

            pair_labels = z.float().to(codes.device)
            soft_a = torch.softmax(logits, dim=-1)
            soft_b = torch.softmax(logits_b, dim=-1)
            pairwise_loss = self.contrastive_loss(
                soft_a, soft_b, pair_labels, margin=self.hparams.contrastive_margin
            )
            self.log("cluster_net_train/train/pairwise_loss", pairwise_loss, on_epoch=True)
            self._epoch_pairwise_losses.append(pairwise_loss.detach().item())
            loss = loss + self.hparams.contrastive_weight * pairwise_loss
        else:
            logits_b = None

        # ---- subcluster net update ----
        if not self.hparams.ignore_subclusters and optimizer_idx == self.optimizers_dict_idx["subcluster_net_opt"]:
            logits_detached = logits.detach()
            if self.hparams.start_sub_clustering <= self.current_epoch:
                sublogits = self.subcluster(codes, logits_detached)
                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                    codes, logits_detached, sublogits, self.K, self.n_sub, self.mus_sub,
                    covs_sub=self.covs_sub if self.hparams.subcluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                    pis_sub=self.pi_sub,
                )
                self._epoch_subcluster_losses.append(subcluster_loss.detach().item())
                loss = self.hparams.subcluster_loss_weight * subcluster_loss

                if codes_b is not None and z is not None and self.hparams.subcluster_contrastive_weight > 0 and logits_b is not None:
                    logits_b_detached = logits_b.detach()
                    sublogits_b = self.subcluster(codes_b, logits_b_detached)
                    pair_labels = z.float().to(codes.device)

                    # Only compare pairs that currently share a predicted
                    # top-level cluster — comparing across different
                    # predicted clusters is a trivial, gradient-free
                    # comparison (disjoint support => always maximally far).
                    same_pred_cluster = (logits_detached.argmax(-1) == logits_b_detached.argmax(-1))

                    if same_pred_cluster.any():
                        sub_pairwise_loss = self.contrastive_loss(
                            sublogits[same_pred_cluster],
                            sublogits_b[same_pred_cluster],
                            pair_labels[same_pred_cluster],
                            margin=self.hparams.subcluster_contrastive_margin,
                        )
                    else:
                        sub_pairwise_loss = torch.zeros((), device=codes.device)
                    self.log("cluster_net_train/train/sub_pairwise_loss", sub_pairwise_loss, on_epoch=True)
                    self._epoch_sub_pairwise_losses.append(sub_pairwise_loss.detach().item())
                    loss = loss + self.hparams.subcluster_contrastive_weight * sub_pairwise_loss
            else:
                sublogits = None
                loss = None
        else:
            sublogits = None

        # log data only once
        if optimizer_idx == len(self.optimizers_dict_idx) - 1:
            (
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
            ) = self.training_utils.log_codes_and_responses(
                self.codes,
                self.train_gt,
                self.train_resp,
                self.train_resp_sub,
                codes,
                logits.detach() if logits is not None else None,
                y,
                sublogits=sublogits,
            )

        return loss if loss is not None else None

    def validation_step(self, batch, batch_idx):
        if batch[0].ndim == 3:
            x, y, z = batch
        else:
            x, y = batch

        codes_a, _ = self._extract_codes(x)
        logits = self.cluster_net(codes_a)
        if batch_idx == 0 and (self.current_epoch < 5 or self.current_epoch % 50 == 0):
            self.log_logits(logits)

        if self.current_training_stage != "gather_codes":
            cluster_loss = self.training_utils.cluster_loss_function(
                codes_a.view(-1, self.codes_dim),
                logits,
                model_mus=self.mus,
                K=self.K,
                codes_dim=self.codes_dim,
                model_covs=self.covs if self.hparams.cluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                pi=self.pi,
            )
            loss = self.hparams.cluster_loss_weight * cluster_loss
            self.log("cluster_net_train/val/cluster_loss", loss)

            if self.current_epoch >= self.hparams.start_sub_clustering and not self.hparams.ignore_subclusters:
                subclusters = self.subcluster(codes_a, logits)
                subcluster_loss = self.training_utils.subcluster_loss_function_new(
                    codes_a.view(-1, self.codes_dim),
                    logits,
                    subclusters,
                    self.K,
                    self.n_sub,
                    self.mus_sub,
                    covs_sub=self.covs_sub if self.hparams.subcluster_loss in ("diag_NIG", "KL_GMM_2") else None,
                    pis_sub=self.pi_sub,
                )
                self.log("cluster_net_train/val/subcluster_loss", subcluster_loss)
                loss += self.hparams.subcluster_loss_weight * subcluster_loss
            else:
                subclusters = None
        else:
            loss = torch.tensor(1.0)
            subclusters = None
            logits = None

        (
            self.codes,
            self.val_gt,
            self.val_resp,
            self.val_resp_sub,
        ) = self.training_utils.log_codes_and_responses(
            self.codes,
            self.val_gt,
            self.val_resp,
            model_resp_sub=self.val_resp_sub,
            codes=codes_a,
            logits=logits,
            y=y,
            sublogits=subclusters,
            stage="val",
        )

        return {"loss": loss}

    def training_epoch_end(self, outputs):
        """Perform logging operations and computes the clusters' and the subclusters' centers.
        Also perform split and merges steps.

        NOTE: mus/covs/pi are recomputed on the ORIGINAL unmodified schedule
        (gated only by `freeze_mus`, never by whether contrastive supervision
        is active). This is the fix for the bug where an always-on gate
        (`not contrastive_only`) permanently froze the GMM fit.
        """

        if self.current_training_stage == "gather_codes":
            self.plot_utils = PlotUtils(
                self.hparams, self.logger, self.codes.view(-1, self.codes_dim)
            )
            self.prior.init_priors(self.codes.view(-1, self.codes_dim))
            if self.centers is not None:
                self.mus = torch.from_numpy(self.centers).cpu()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init
            else:
                self.freeze_mus_after_init_until = 0
                self.mus, self.covs, self.pi, init_labels = init_mus_and_covs(
                    codes=self.codes.view(-1, self.codes_dim),
                    K=self.K,
                    how_to_init_mu=self.hparams.how_to_init_mu,
                    logits=self.train_resp,
                    use_priors=self.hparams.use_priors,
                    prior=self.prior,
                    random_state=0,
                    device=self.device,
                )
                if self.hparams.use_labels_for_eval:
                    if (self.train_gt < 0).any():
                        gt = self.train_gt[self.train_gt > -1]
                        init_labels = init_labels[self.train_gt > -1]
                    else:
                        gt = self.train_gt
                    if len(gt) > 2 * (10 ** 5):
                        gt = gt[:2 * (10 ** 5)]
                    init_nmi = normalized_mutual_info_score(gt, init_labels)
                    init_ari = adjusted_rand_score(gt, init_labels)
                    self.log("cluster_net_train/init_nmi", init_nmi)
                    self.log("cluster_net_train/init_ari", init_ari)
                if self.hparams.log_emb == "every_n_epochs" and (self.current_epoch % self.hparams.log_emb_every == 0 or self.current_epoch == 1):
                    self.plot_utils.visualize_embeddings(
                        self.hparams, self.logger, self.codes_dim,
                        vae_means=self.codes, vae_labels=init_labels, val_resp=None,
                        current_epoch=self.current_epoch, y_hat=None, centers=self.mus,
                        stage="init_Kmeans",
                    )

        else:
            if not self.hparams.ignore_subclusters:
                clus_losses, subclus_losses = outputs[0], outputs[1]
            else:
                clus_losses = outputs
            avg_clus_loss = torch.stack([x["loss"] for x in clus_losses]).mean()
            self.log("cluster_net_train/train/avg_cluster_loss", avg_clus_loss)
            if self.current_epoch >= self.hparams.start_sub_clustering and not self.hparams.ignore_subclusters:
                avg_subclus_loss = torch.stack([x["loss"] for x in subclus_losses]).mean()
                self.log("cluster_net_train/train/avg_subcluster_loss", avg_subclus_loss)

            perform_split = self.training_utils.should_perform_split(self.current_epoch) and self.centers is None
            perform_merge = self.training_utils.should_perform_merge(self.current_epoch, self.split_performed) and self.centers is None

            if self.centers is not None:
                self.mus = torch.from_numpy(self.centers).cpu()
                self.centers = None
                self.init_covs_and_pis_given_mus()
                self.freeze_mus_after_init_until = self.current_epoch + self.hparams.freeze_mus_after_init

            freeze_mus = self.training_utils.freeze_mus(self.current_epoch, self.split_performed) or self.current_epoch <= self.freeze_mus_after_init_until

            if not freeze_mus:
                (
                    self.pi,
                    self.mus,
                    self.covs,
                ) = self.training_utils.comp_cluster_params(
                    self.train_resp, self.codes.view(-1, self.codes_dim), self.pi, self.K, self.prior,
                )

            if (self.hparams.start_sub_clustering == self.current_epoch + 1) or (self.hparams.ignore_subclusters and (perform_split or perform_merge)):
                (
                    self.pi_sub,
                    self.mus_sub,
                    self.covs_sub,
                ) = self.training_utils.init_subcluster_params(
                    self.train_resp, self.train_resp_sub, self.codes.view(-1, self.codes_dim), self.K, self.n_sub, self.prior,
                )
            elif self.hparams.start_sub_clustering <= self.current_epoch and not freeze_mus and not self.hparams.ignore_subclusters:
                # Empty subclusters can transiently occur; skip the update
                # rather than crashing the run (kept from the earlier fix).
                try:
                    (
                        self.pi_sub,
                        self.mus_sub,
                        self.covs_sub,
                    ) = self.training_utils.comp_subcluster_params(
                        self.train_resp, self.train_resp_sub, self.codes, self.K, self.n_sub,
                        self.mus_sub, self.covs_sub, self.pi_sub, self.prior,
                    )
                except IndexError as e:
                    print(
                        f"[Epoch {self.current_epoch}] Skipping subcluster param "
                        f"update — a subcluster was empty this epoch ({e}). "
                        f"Keeping previous mus_sub/covs_sub/pi_sub."
                    )

            if perform_split and not freeze_mus:
                self.training_utils.last_performed = "split"
                split_decisions = split_step(
                    self.K, self.codes, self.train_resp, self.train_resp_sub,
                    self.mus, self.mus_sub, self.hparams.cov_const, self.hparams.alpha,
                    self.hparams.split_prob, self.prior, self.hparams.ignore_subclusters,
                )
                if split_decisions.any():
                    self.split_performed = True
                    self.perform_split_operations(split_decisions)

            if perform_merge and not freeze_mus:
                self.training_utils.last_performed = "merge"
                mus_to_merge, highest_ll_mus = merge_step(
                    self.mus, self.train_resp, self.codes, self.K,
                    self.hparams.raise_merge_proposals, self.hparams.cov_const,
                    self.hparams.alpha, self.hparams.merge_prob, prior=self.prior,
                )
                if len(mus_to_merge) > 0:
                    self.merge_performed = True
                    self.perform_merge(mus_to_merge, highest_ll_mus)

            if self.hparams.log_metrics_at_train and self.hparams.evaluate_every_n_epochs > 0 and self.current_epoch % self.hparams.evaluate_every_n_epochs == 0:
                self.log_clustering_metrics()

            with torch.no_grad():
                if self.hparams.log_emb == "every_n_epochs" and (self.current_epoch % self.hparams.log_emb_every == 0 or self.current_epoch < 2):
                    self.plot_histograms()
                    self.plot_utils.visualize_embeddings(
                        self.hparams, self.logger, self.codes_dim,
                        vae_means=self.codes,
                        vae_labels=None if not self.hparams.use_labels_for_eval else self.train_gt,
                        val_resp=self.train_resp, current_epoch=self.current_epoch, y_hat=None,
                        centers=self.mus, training_stage='train',
                    )
                    if self.hparams.dataset == "synthetic":
                        if self.split_performed or self.merge_performed:
                            self.plot_utils.update_colors(self.split_performed, self.mus_ind_to_split, self.mus_inds_to_merge)
                        elif self.hparams.use_labels_for_eval:
                            self.plot_utils.plot_cluster_and_decision_boundaries(
                                samples=self.codes, labels=self.train_resp.argmax(-1), gt_labels=self.train_gt,
                                net_centers=self.mus, net_covs=self.covs, n_epoch=self.current_epoch, cluster_net=self,
                            )
                    if self.current_epoch in (0, 1, 2, 3, 4, 5, 10, 100, 200, 300, 400, 500, self.hparams.start_sub_clustering, self.hparams.start_sub_clustering + 1) or self.split_performed or self.merge_performed:
                        self.plot_histograms(for_thesis=True)

        if self.split_performed or self.merge_performed:
            self.update_params_split_merge()
            action = "split" if self.split_performed else "merge"
            print(f"[{action.capitalize()}] K updated → {self.K} clusters")

            avg_cluster_loss = float(np.mean(self._epoch_cluster_losses)) if self._epoch_cluster_losses else float("nan")
            avg_subcluster_loss = float(np.mean(self._epoch_subcluster_losses)) if self._epoch_subcluster_losses else float("nan")
            avg_pairwise_loss = float(np.mean(self._epoch_pairwise_losses)) if self._epoch_pairwise_losses else float("nan")
            avg_sub_pairwise_loss = float(np.mean(self._epoch_sub_pairwise_losses)) if self._epoch_sub_pairwise_losses else float("nan")

            loss_parts = [
                f"cluster_loss={avg_cluster_loss:.4f}",
                f"subcluster_loss={avg_subcluster_loss:.4f}",
            ]
            if self._epoch_pairwise_losses:
                loss_parts.append(f"pairwise_loss={avg_pairwise_loss:.4f}")
            if self._epoch_sub_pairwise_losses:
                loss_parts.append(f"sub_pairwise_loss={avg_sub_pairwise_loss:.4f}")
            print(f"  [Epoch {self.current_epoch}] losses at {action} → " + ", ".join(loss_parts))

        # ---- record this epoch's history (skip the code-gathering epoch) ----
        if self.current_training_stage != "gather_codes":
            self.history["epoch"].append(self.current_epoch)
            self.history["K"].append(self.K)
            self.history["cluster_loss"].append(float(np.mean(self._epoch_cluster_losses)) if self._epoch_cluster_losses else np.nan)
            self.history["subcluster_loss"].append(float(np.mean(self._epoch_subcluster_losses)) if self._epoch_subcluster_losses else np.nan)
            self.history["pairwise_loss"].append(float(np.mean(self._epoch_pairwise_losses)) if self._epoch_pairwise_losses else np.nan)
            self.history["sub_pairwise_loss"].append(float(np.mean(self._epoch_sub_pairwise_losses)) if self._epoch_sub_pairwise_losses else np.nan)

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("cluster_net_train/val/avg_val_loss", avg_loss)
        if self.current_training_stage != "gather_codes" and self.hparams.evaluate_every_n_epochs and self.current_epoch % self.hparams.evaluate_every_n_epochs == 0:
            z = self.val_resp.argmax(axis=1).cpu()
            nmi = normalized_mutual_info_score(self.val_gt, z)
            self.last_val_NMI = nmi
            self.log_clustering_metrics(stage="val")
            if not (self.split_performed or self.merge_performed) and self.hparams.log_metrics_at_train:
                self.log_clustering_metrics(stage="total")

        if self.hparams.log_emb == "every_n_epochs" and self.current_epoch % 10 == 0 and len(self.val_gt) > 10:
            self.plot_utils.visualize_embeddings(
                self.hparams, self.logger, self.codes_dim,
                vae_means=self.codes, vae_labels=self.val_gt,
                val_resp=self.val_resp if self.val_resp != [] else None,
                current_epoch=self.current_epoch, y_hat=None, centers=None,
                training_stage="val_thesis",
            )

        if self.current_epoch > self.hparams.start_sub_clustering and (self.current_epoch % 50 == 0 or self.current_epoch == self.hparams.train_cluster_net - 1):
            from pytorch_lightning.loggers.base import DummyLogger
            if not isinstance(self.logger, DummyLogger):
                self.plot_histograms(train=False, for_thesis=True)

    def subcluster(self, codes, logits, hard_assignment=True):
        sub_clus_resp = self.subclustering_net(codes)  # unnormalized
        z = logits.argmax(-1)

        mask = torch.zeros_like(sub_clus_resp)
        mask[np.arange(len(z)), 2 * z] = 1.
        mask[np.arange(len(z)), 2 * z + 1] = 1.

        sub_clus_resp = torch.nn.functional.softmax(
            sub_clus_resp.masked_fill((1 - mask).bool(), float('-inf')) * self.subclustering_net.softmax_norm, dim=1
        )
        return sub_clus_resp

    def update_subcluster_net_split(self, split_decisions):
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_split(split_decisions, self.hparams.split_init_weights_sub)
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())

    def perform_split_operations(self, split_decisions):
        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            clus_opt = self.optimizers()

        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)
        self.cluster_net.update_K_split(split_decisions, self.hparams.init_new_weights, self.subclustering_net)
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        mus_ind_to_split = torch.nonzero(torch.tensor(split_decisions), as_tuple=False)
        (
            self.mus_new, self.covs_new, self.pi_new,
            self.mus_sub_new, self.covs_sub_new, self.pi_sub_new,
        ) = update_models_parameters_split(
            split_decisions, self.mus, self.covs, self.pi, mus_ind_to_split,
            self.mus_sub, self.covs_sub, self.pi_sub, self.codes, self.train_resp,
            self.train_resp_sub, self.n_sub, self.hparams.how_to_init_mu_sub,
            self.prior, use_priors=self.hparams.use_priors,
        )
        print(f"Splitting clusters {np.arange(self.K)[split_decisions.bool().tolist()]}")
        self.K += len(mus_ind_to_split)

        if not self.hparams.ignore_subclusters:
            self.update_subcluster_net_split(split_decisions)
        self.mus_ind_to_split = mus_ind_to_split

    def update_subcluster_nets_merge(self, merge_decisions, pairs_to_merge, highest_ll):
        subclus_opt = self.optimizers()[self.optimizers_dict_idx["subcluster_net_opt"]]
        for p in self.subclustering_net.parameters():
            subclus_opt.state.pop(p)
        self.subclustering_net.update_K_merge(merge_decisions, pairs_to_merge=pairs_to_merge, highest_ll=highest_ll, init_new_weights=self.hparams.merge_init_weights_sub)
        subclus_opt.param_groups[0]["params"] = list(self.subclustering_net.parameters())

    def perform_merge(self, mus_lists_to_merge, highest_ll_mus, use_priors=True):
        print(f"Merging clusters {mus_lists_to_merge}")
        mus_lists_to_merge = torch.tensor(mus_lists_to_merge)
        inds_to_mask = torch.zeros(self.K, dtype=bool)
        inds_to_mask[mus_lists_to_merge.flatten()] = 1
        (
            self.mus_new, self.covs_new, self.pi_new,
            self.mus_sub_new, self.covs_sub_new, self.pi_sub_new,
        ) = update_models_parameters_merge(
            mus_lists_to_merge, inds_to_mask, self.K, self.mus, self.covs, self.pi,
            self.mus_sub, self.covs_sub, self.pi_sub, self.codes, self.train_resp,
            self.prior, use_priors=self.hparams.use_priors, n_sub=self.n_sub,
            how_to_init_mu_sub=self.hparams.how_to_init_mu_sub,
        )
        self.K -= len(highest_ll_mus)

        if not self.hparams.ignore_subclusters:
            self.update_subcluster_nets_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus)

        if not self.hparams.ignore_subclusters:
            clus_opt = self.optimizers()[self.optimizers_dict_idx["cluster_net_opt"]]
        else:
            clus_opt = self.optimizers()

        for p in self.cluster_net.class_fc2.parameters():
            clus_opt.state.pop(p)

        self.cluster_net.update_K_merge(inds_to_mask, mus_lists_to_merge, highest_ll_mus, init_new_weights=self.hparams.init_new_weights)
        clus_opt.param_groups[1]["params"] = list(self.cluster_net.class_fc2.parameters())
        self.cluster_net.class_fc2.to(self._device)
        self.mus_inds_to_merge = mus_lists_to_merge

    def configure_optimizers(self):
        cluster_params = torch.nn.ParameterList([p for n, p in self.cluster_net.named_parameters() if "class_fc2" not in n])
        cluster_net_opt = optim.Adam(cluster_params, lr=self.hparams.cluster_lr)
        cluster_net_opt.add_param_group({"params": self.cluster_net.class_fc2.parameters()})
        self.optimizers_dict_idx = {"cluster_net_opt": 0}

        if self.hparams.lr_scheduler == "StepLR":
            cluster_scheduler = torch.optim.lr_scheduler.StepLR(cluster_net_opt, step_size=20)
        elif self.hparams.lr_scheduler == "ReduceOnP":
            cluster_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(cluster_net_opt, mode="min", factor=0.5, patience=4)
        else:
            cluster_scheduler = None

        if not self.hparams.ignore_subclusters:
            sub_clus_opt = optim.Adam(self.subclustering_net.parameters(), lr=self.hparams.subcluster_lr)
            self.optimizers_dict_idx["subcluster_net_opt"] = 1
            return (
                {"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"},
                {"optimizer": sub_clus_opt},
            )
        return {"optimizer": cluster_net_opt, "scheduler": cluster_scheduler, "monitor": "cluster_net_train/val/cluster_loss"} if cluster_scheduler else cluster_net_opt

    def update_params_split_merge(self):
        self.mus = self.mus_new
        self.covs = self.covs_new
        self.mus_sub = self.mus_sub_new
        self.covs_sub = self.covs_sub_new
        self.pi = self.pi_new
        self.pi_sub = self.pi_sub_new

    def init_covs_and_pis_given_mus(self):
        if self.hparams.use_priors_for_net_params_init:
            _, cov_prior = self.prior.init_priors(self.mus)
            self.covs = torch.stack([cov_prior for k in range(self.K)])
            p_counts = torch.ones(self.K) * 10
            self.pi = p_counts / float(self.K * 10)
        else:
            dis_mat = torch.empty((len(self.codes), self.K))
            for i in range(self.K):
                dis_mat[:, i] = torch.sqrt(((self.codes - self.mus[i]) ** 2).sum(axis=1))
            hard_assign = torch.argmin(dis_mat, dim=1)

            vals, counts = torch.unique(hard_assign, return_counts=True)
            if len(counts) < self.K:
                new_counts = []
                for k in range(self.K):
                    if k in vals:
                        new_counts.append(counts[vals == k])
                    else:
                        new_counts.append(0)
                counts = torch.tensor(new_counts)
            pi = counts / float(len(self.codes))
            data_covs = compute_data_covs_hard_assignment(hard_assign.numpy(), self.codes, self.K, self.mus.cpu(), self.prior)
            if self.use_priors:
                covs = []
                for k in range(self.K):
                    codes_k = self.codes[hard_assign == k]
                    cov_k = self.prior.compute_post_cov(counts[k], codes_k.mean(axis=0), data_covs[k])
                    covs.append(cov_k)
                covs = torch.stack(covs)
            self.covs = covs
            self.pi = pi

    def log_logits(self, logits):
        for k in range(self.K):
            max_k = logits[logits.argmax(axis=1) == k].detach().cpu().numpy()
            if len(max_k > 0):
                fig = plt.figure(figsize=(10, 3))
                for i in range(len(max_k[:20])):
                    if i == 0:
                        plt.bar(np.arange(self.K), max_k[i], fill=False, label=len(max_k))
                    else:
                        plt.bar(np.arange(self.K), max_k[i], fill=False)
                plt.xlabel("Clusters inds")
                plt.ylabel("Softmax histogram")
                plt.title(f"Epoch {self.current_epoch}: cluster {k}")
                plt.legend()
                plt.close(fig)

    def plot_histograms(self, train=True, for_thesis=False):
        pi = self.pi_new if self.split_performed or self.merge_performed else self.pi
        if self.hparams.ignore_subclusters:
            pi_sub = None
        else:
            pi_sub = (
                self.pi_sub_new if self.split_performed or self.merge_performed
                else self.pi_sub if self.hparams.start_sub_clustering <= self.current_epoch
                else None
            )

        fig = self.plot_utils.plot_weights_histograms(
            K=self.K, pi=pi, start_sub_clustering=self.hparams.start_sub_clustering,
            current_epoch=self.current_epoch, pi_sub=pi_sub, for_thesis=for_thesis,
        )
        stage = "val_for_thesis" if for_thesis else ("train" if train else "val")

        from pytorch_lightning.loggers.base import DummyLogger
        if not isinstance(self.logger, DummyLogger):
            self.logger.experiment.add_figure(f"cluster_net_train/{stage}/clusters_weights_fig", fig, global_step=self.current_epoch)

    def plot_clusters_high_dim(self, stage="train"):
        resps = {"train": (self.train_resp, self.train_resp_sub), "val": (self.val_resp, self.val_resp_sub)}
        gt = {"train": self.train_gt, "val": self.val_gt}
        (resp, resp_sub) = resps[stage]
        cluster_net_labels = self.training_utils.update_labels_after_split_merge(
            resp.argmax(-1), self.split_performed, self.merge_performed,
            self.mus, self.mus_ind_to_split, self.mus_inds_to_merge, resp_sub,
        )
        fig = self.plot_utils.plot_clusters_colored_by_label(
            samples=self.codes, y_gt=gt[stage], n_epoch=self.current_epoch, K=len(torch.unique(gt[stage])),
        )
        plt.close(fig)
        self.logger.log_image(f"cluster_net_train/{stage}/clusters_fig_gt_labels", fig)
        fig = self.plot_utils.plot_clusters_colored_by_net(
            samples=self.codes, y_net=cluster_net_labels, n_epoch=self.current_epoch, K=len(torch.unique(cluster_net_labels)),
        )
        self.logger.log_image("cluster_net_train/train/clusters_fig_net_labels", fig)
        plt.close(fig)

    def log_clustering_metrics(self, stage="train"):
        print("Evaluating...")
        if stage == "train":
            gt = self.train_gt
            resp = self.train_resp
        elif stage == "val":
            gt = self.val_gt
            resp = self.val_resp
            self.log("cluster_net_train/Networks_k", self.K)
        elif stage == "total":
            gt = torch.cat([self.train_gt, self.val_gt])
            resp = torch.cat([self.train_resp, self.val_resp])

        z = resp.argmax(axis=1).cpu()
        unique_z = len(np.unique(z))
        if len(np.unique(z)) >= 5:
            val, z_top5 = torch.topk(resp, k=5, largest=True)
        else:
            z_top5 = None
        if (gt < 0).any():
            z = z[gt > -1]
            z_top5 = z_top5[gt > -1]
            gt = gt[gt > -1]

        gt_nmi = normalized_mutual_info_score(gt, z)
        ari = adjusted_rand_score(gt, z)
        acc_top5, acc = training_utils.cluster_acc(gt, z, z_top5)

        self.log(f"cluster_net_train/{stage}/{stage}_nmi", gt_nmi, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_ari", ari, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc", acc, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/{stage}_acc_top5", acc_top5, on_epoch=True, on_step=False)
        self.log(f"cluster_net_train/{stage}/unique_z", unique_z, on_epoch=True, on_step=False)

        if self.hparams.offline and ((self.hparams.log_metrics_at_train and stage == "train") or (not self.hparams.log_metrics_at_train and stage != "train")):
            print(f"NMI : {gt_nmi}, ARI: {ari}, ACC: {acc}, current K: {unique_z}")
        if self.current_epoch % 10 == 0 and self.current_epoch > 45:
            print(f"Epoch {self.current_epoch} | NMI: {gt_nmi:.4f}, ARI: {ari:.4f}, ACC: {acc:.4f}, current K: {unique_z}")
            cm = confusion_matrix(gt.cpu().numpy(), z.cpu().numpy())
            print("Confusion Matrix:")
            print(cm)

        if self.current_epoch in (0, 1, self.hparams.train_cluster_net - 1):
            alt_stage = "start" if self.current_epoch == 1 or self.hparams.train_cluster_net % self.current_epoch == 0 else "end"
            if unique_z > 1:
                try:
                    silhouette = silhouette_score(self.codes.cpu(), z.cpu().numpy())
                except Exception:
                    silhouette = 0
            else:
                silhouette = 0
            ami = adjusted_mutual_info_score(gt.numpy(), z.numpy())
            (homogeneity, completeness, v_measure) = homogeneity_completeness_v_measure(gt.numpy(), z.numpy())

            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_nmi", gt_nmi, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_ari", ari, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_acc", acc, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_acc_top5", acc_top5, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_silhouette_score", silhouette, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_ami", ami, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_homogeneity", homogeneity, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_v_measure", v_measure, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_{stage}_completeness", completeness, on_epoch=True, on_step=False)
            self.log(f"cluster_net_train/{stage}/alt_{alt_stage}_unique_z", unique_z, on_epoch=True, on_step=False)

    def plot_training_history(self, save_path="logs/training_history.png"):
        """Plot cluster loss, subcluster loss, pairwise loss, and K each in
        their own subplot. Call this after trainer.fit() completes.
        """
        import os
        epochs = self.history["epoch"]
        if len(epochs) == 0:
            print("No training history recorded — nothing to plot.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0, 0]
        ax.plot(epochs, self.history["cluster_loss"], color="tab:blue")
        ax.set_title("Cluster loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(epochs, self.history["subcluster_loss"], color="tab:orange")
        ax.set_title("Subcluster loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(epochs, self.history["pairwise_loss"], color="tab:green", label="pairwise loss")
        if any(not np.isnan(v) for v in self.history["sub_pairwise_loss"]):
            ax.plot(epochs, self.history["sub_pairwise_loss"], color="tab:red", label="sub-pairwise loss")
            ax.legend()
        ax.set_title("Pairwise loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.step(epochs, self.history["K"], where="post", color="black", linewidth=2)
        ax.set_title("Number of clusters (K)")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("K")
        ax.grid(alpha=0.3)

        plt.suptitle("Training history", fontsize=14, y=1.02)
        plt.tight_layout()

        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Training history plot saved to: {save_path}")
        return fig

    @staticmethod
    def add_model_specific_args(parent_parser):
        # NOTE: defaults below match the ORIGINAL DeepDPM values
        # (cluster_loss_weight=1, train_cluster_net=300) — not the 0.7/200
        # values that had drifted in in an earlier iteration.
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--init_k", default=3, type=int, help="number of initial clusters")
        parser.add_argument("--clusternet_hidden", type=int, default=50, help="The dimensions of the hidden dim of the clusternet. Defaults to 50.")
        parser.add_argument("--clusternet_hidden_layer_list", type=int, nargs="+", default=[50], help="The hidden layers in the clusternet. Defaults to [50, 50].")
        parser.add_argument("--transform_input_data", type=str, default="normalize", choices=["normalize", "min_max", "standard", "standard_normalize", "None", None], help="Use normalization for embedded data")
        parser.add_argument("--cluster_loss_weight", type=float, default=1)
        parser.add_argument("--init_cluster_net_weights", action="store_true", default=False)
        parser.add_argument("--when_to_compute_mu", type=str, choices=["once", "every_epoch", "every_5_epochs"], default="every_epoch")
        parser.add_argument("--how_to_compute_mu", type=str, choices=["kmeans", "soft_assign"], default="soft_assign")
        parser.add_argument("--how_to_init_mu", type=str, choices=["kmeans", "soft_assign", "kmeans_1d"], default="kmeans")
        parser.add_argument("--how_to_init_mu_sub", type=str, choices=["kmeans", "soft_assign", "kmeans_1d"], default="kmeans_1d")
        parser.add_argument("--log_emb_every", type=int, default=20)
        parser.add_argument("--log_emb", type=str, default="never", choices=["every_n_epochs", "only_sampled", "never"])
        parser.add_argument("--train_cluster_net", type=int, default=300, help="Number of epochs to pretrain the cluster net")
        parser.add_argument("--cluster_lr", type=float, default=0.0005)
        parser.add_argument("--subcluster_lr", type=float, default=0.005)
        parser.add_argument("--lr_scheduler", type=str, default="ReduceOnP", choices=["StepLR", "None", "ReduceOnP"])
        parser.add_argument("--start_sub_clustering", type=int, default=35)
        parser.add_argument("--subcluster_loss_weight", type=float, default=1.0)
        parser.add_argument("--start_splitting", type=int, default=45)
        parser.add_argument("--alpha", type=float, default=10.0)
        parser.add_argument("--softmax_norm", type=float, default=1)
        parser.add_argument("--subcluster_softmax_norm", type=float, default=1)
        parser.add_argument("--split_prob", type=float, default=None, help="Split with this probability even if split rule is not met. If None, uses min(1,H).")
        parser.add_argument("--merge_prob", type=float, default=None, help="Merge with this probability even if merge rule is not met. If None, uses min(1,H).")
        parser.add_argument("--init_new_weights", type=str, default="same", choices=["same", "random", "subclusters"], help="How to create new weights after split.")
        parser.add_argument("--start_merging", type=int, default=45, help="The epoch in which to start consider merge proposals")
        parser.add_argument("--merge_init_weights_sub", type=str, default="highest_ll", help="How to initialize the weights of the subclusters of the merged clusters.")
        parser.add_argument("--split_init_weights_sub", type=str, default="random", choices=["same_w_noise", "same", "random"], help="How to initialize the weights of the subclusters of the split clusters.")
        parser.add_argument("--split_every_n_epochs", type=int, default=10)
        parser.add_argument("--split_merge_every_n_epochs", type=int, default=30)
        parser.add_argument("--merge_every_n_epochs", type=int, default=10)
        parser.add_argument("--raise_merge_proposals", type=str, default="brute_force_NN", help="how to raise merge proposals")
        parser.add_argument("--cov_const", type=float, default=0.005, help="gmms covs (in the Hastings ratio) will be torch.eye * cov_const")
        parser.add_argument("--freeze_mus_submus_after_splitmerge", type=int, default=2, help="Epochs to freeze mus/sub-mus following a split or merge step")
        parser.add_argument("--freeze_mus_after_init", type=int, default=5, help="Epochs to freeze mus/sub-mus following a new initialization")
        parser.add_argument("--use_priors", type=int, default=1, help="Whether to use priors when computing model's parameters")
        parser.add_argument("--prior", type=str, default="NIW", choices=["NIW", "NIG"])
        parser.add_argument("--pi_prior", type=str, default="uniform", choices=["uniform", None])
        parser.add_argument("--prior_dir_counts", type=float, default=0.1)
        parser.add_argument("--prior_kappa", type=float, default=0.0001)
        parser.add_argument("--NIW_prior_nu", type=float, default=None, help="Need to be at least codes_dim + 1")
        parser.add_argument("--prior_mu_0", type=str, default="data_mean")
        parser.add_argument("--prior_sigma_choice", type=str, default="isotropic", choices=["iso_005", "iso_001", "iso_0001", "data_std"])
        parser.add_argument("--prior_sigma_scale", type=float, default=".005")
        parser.add_argument("--prior_sigma_scale_step", type=float, default=1., help="add to change sigma scale between alternations")
        parser.add_argument("--compute_params_every", type=int, default=1, help="How frequently to compute the clustering params (mus, sub, pis)")
        parser.add_argument("--start_computing_params", type=int, default=25, help="When to start to compute the clustering params (mus, sub, pis)")
        parser.add_argument("--cluster_loss", type=str, default="KL_GMM_2", choices=["diag_NIG", "isotropic", "isotropic_2", "isotropic_3", "isotropic_4", "KL_GMM_2"])
        parser.add_argument("--subcluster_loss", type=str, default="isotropic", choices=["diag_NIG", "isotropic", "KL_GMM_2"])
        parser.add_argument("--use_priors_for_net_params_init", type=bool, default=True, help="If centers are given at re-init, use priors (True) or min-dist (False) for covs/pis.")
        parser.add_argument("--ignore_subclusters", type=bool, default=False)
        parser.add_argument("--log_metrics_at_train", type=bool, default=False)
        parser.add_argument("--evaluate_every_n_epochs", type=int, default=5, help="How often to evaluate the net")

        # ---- pairwise-supervision extension (new, off by default) ----
        parser.add_argument("--contrastive_weight", type=float, default=0.0, help="Weight of the auxiliary pairwise contrastive term (top-level clusters). 0 = pure DeepDPM.")
        parser.add_argument("--contrastive_margin", type=float, default=1.0, help="Margin for the top-level pairwise contrastive term.")
        parser.add_argument("--subcluster_contrastive_weight", type=float, default=0.0, help="Weight of the auxiliary pairwise contrastive term at the subcluster level.")
        parser.add_argument("--subcluster_contrastive_margin", type=float, default=1.0, help="Margin for the subcluster-level pairwise term. Kept separate from --contrastive_margin since the sub-level comparison space (2-dim simplex, max distance sqrt(2)) doesn't shrink as K grows.")

        return parser