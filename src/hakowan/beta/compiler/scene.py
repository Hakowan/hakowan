from dataclasses import dataclass, field

from .view import View

@dataclass
class Scene:
    views: list[View] = field(default_factory=list)

    def append(self, view: View) -> "Scene":
        self.views.append(view)
        return self
