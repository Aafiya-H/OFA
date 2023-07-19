from dataclasses import dataclass
from ..ofa_task import OFAConfig, OFATask
from fairseq.tasks import register_task

@dataclass
class VisualStorytellingConfig(OFAConfig):
    pass


@register_task("visual_storytelling",dataclass=VisualStorytellingConfig)
class VisualStorytelling(OFATask):
    pass

