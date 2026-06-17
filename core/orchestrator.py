from pathlib import Path

from core.agent_loader import AgentLoader
from core.config import load_config
from core.llm_client import VLLMClient

from skills.frame_sampler import sample_frames
from skills.frame_encoder import encode_frame
from skills.timeline import Timeline
from skills.report_generator import save_report


class VideoOrchestrator:

    def __init__(
        self,
        video_path,
        sample_interval=0.5,
        stream_mode=False,
        report_only=False
    ):

        self.video_path = video_path
        self.sample_interval = sample_interval
        self.stream_mode = stream_mode
        self.report_only = report_only

    def run(self):

        cfg = load_config()

        agents = AgentLoader().load()

        client = VLLMClient(
            cfg["vllm_endpoint"],
            cfg["model"]
        )

        timeline = Timeline()

        scene_prompt = agents["scene_detector"]

        for timestamp, frame in sample_frames(
            self.video_path,
            self.sample_interval
        ):

            image_b64 = encode_frame(frame)

            result = client.ask(
                scene_prompt,
                image_b64
            )

            print(
                f"[{timestamp:.1f}s] "
                f"{result[:120]}"
            )

            timeline.add({
                "time": timestamp,
                "result": result
            })

        summary_prompt = agents["summary_agent"]

        final_summary = client.ask(
            f"{summary_prompt}\n\n"
            f"{timeline.events}"
        )

        report = save_report(
            final_summary,
            Path(self.video_path).stem
        )

        print()
        print("Analysis Complete")
        print(report)