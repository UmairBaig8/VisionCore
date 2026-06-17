from pathlib import Path


class SkillRegistry:

    def __init__(self):

        self.skills = []

    def load(self):

        for file in Path("skills").glob("*.py"):

            if file.name.startswith("_"):
                continue

            self.skills.append(file.stem)

        return self.skills