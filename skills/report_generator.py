from pathlib import Path


def save_report(text, video_name):

    output = Path("output/reports")

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path = output / f"{video_name}.md"

    report_path.write_text(text)

    return report_path