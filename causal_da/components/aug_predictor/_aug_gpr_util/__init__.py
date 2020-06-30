from packaging import version
import sklearn

if version.parse(sklearn.__version__) <= version.parse('0.21'):
    from .aug_gpr_upto_v0_21 import _MLOriginalTargetDataAugGPR as MLOriginalDataAugGPR
else:
    from .aug_gpr_from_v0_21 import _MLOriginalTargetDataAugGPR as MLOriginalDataAugGPR
