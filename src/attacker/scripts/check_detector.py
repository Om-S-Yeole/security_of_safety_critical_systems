from pathlib import Path

from src.attacker.scripts.detector import YOLOv8Detector

MODEL_PATH = Path("detector", "weights", "best.pt")

IMAGE_PATH = Path(
    "project_data",
    "balanced_split",
    "train",
    "images",
    "aircraft-1-_jpg.rf.8a4d3c9f141b580393f093f67ee5b3fb_brightness_1.jpg",
)


def main():
    detector = YOLOv8Detector(
        model_path=str(MODEL_PATH),
        image_size=640,
        confidence_threshold=0.60,
        iou_threshold=0.70,
        device="cuda:0",
    )

    predictions = detector.predict(str(IMAGE_PATH))

    for prediction in predictions:
        print("\nImage:", prediction["image_path"])

        for box, score, class_name in zip(
            prediction["boxes"],
            prediction["scores"],
            prediction["class_names"],
        ):
            print(f"Class={class_name:12s} Confidence={score:.3f} Box={box}")

        if not prediction["boxes"]:
            print("No detections above the confidence threshold.")


if __name__ == "__main__":
    main()
