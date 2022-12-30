from dataclasses import dataclass, field
from typing import Dict
import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas

class_translator = {
    "ocr": nac.OcrAug,
    "contextual_w_e": naw.ContextualWordEmbsAug,
    "synonym": naw.SynonymAug,
    "backtranslation": naw.BackTranslationAug,
    "contextual_s_e": nas.ContextualWordEmbsForSentenceAug,
    "abstractive_summ": nas.AbstSummAug,
}


@dataclass
class NLPAugConfig:
    """
    Configuration for augmenters.

    Parameters
    ----------
    name : str
        Name of the data augmentation technique. Possible values currently are `ocr` (for OCR augmentation), `contextual_w_e`
        for Contextual Word Embedding augmentation, `synonym`, `backtranslation`, `contextual_s_e` for Contextual Word Embeddings for Sentence Augmentation,
        `abstractive_summ`.
    proportion : float
        Proportion of data augmentation.
    aug_kwargs : Dict
        Arguments for the data augmentation class. See https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb
    """

    name: str = field(metadata={"help": "Name of the data augmentation technique"})
    proportion: float = field(
        default=0.1, metadata={"help": "proportion of data augmentation"}
    )
    aug_kwargs: Dict = field(
        default=None,
        metadata={
            "help": "Arguments for the data augmentation class. See https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb"
        },
    )
