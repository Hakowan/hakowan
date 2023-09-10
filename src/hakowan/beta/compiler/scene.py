from dataclasses import dataclass, field

from .view import View

@dataclass
class Scene:
    views: list[View] = field(default_factory=list)

    def __len__(self):
        return self.views.__len__()

    def __getitem__(self, key):
        return self.views.__getitem__(key)

    def __setitem__(self, key, value: View):
        return self.views.__setitem__(key, value)

    def __delitem__(self, key):
        return self.views.__delitem__(key)

    def __iter__(self):
        return self.views.__iter__()

    def __contains__(self, item):
        return self.views.__contains__(item)

    def append(self, view: View) -> "Scene":
        self.views.append(view)
        return self
