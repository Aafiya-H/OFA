from dataclasses import dataclass
from ..ofa_task import OFAConfig, OFATask
from fairseq.tasks import register_task

@dataclass
class StoryContinuationConfig(OFAConfig):
    pass


@register_task("story_continuation",dataclass=StoryContinuationConfig)
class StoryContinuationTask(OFATask):
    pass

