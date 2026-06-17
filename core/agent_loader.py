from pathlib import Path


class AgentLoader:

    def __init__(self):

        self.agents = {}

    def load(self):

        agent_dir = Path("agents")

        for file in agent_dir.glob("*.md"):

            self.agents[file.stem] = file.read_text()

        return self.agents