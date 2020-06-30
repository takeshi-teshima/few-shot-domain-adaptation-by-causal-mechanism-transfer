from .partial_loo_krr import PartialLOOCVKRR
from .util import Timer


class AugKRR(PartialLOOCVKRR):
    def fit(self, x, y, augmenter, augment_size):
        with Timer() as t:
            X_aug, Y_aug = augmenter.augment_to_size(
                x,
                y,
                include_original=False,
                with_latent=False,
                size=augment_size,
                with_acceptance_ratio=False)
        print(f'[Augmentation]: {t.time}')
        return super().fit(x, y, X_aug, Y_aug)
