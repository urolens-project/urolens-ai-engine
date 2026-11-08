from pydantic import BaseModel, Field


class Detection(BaseModel):
    class_id: int = Field(description="YOLO class ID")
    class_name: str = Field(
        description="Particle name, with underscores — matches the keys of InferenceResult.particles"
    )
    confidence: float = Field(description="Detection confidence score 0.0 to 1.0")
    bbox: list[float] = Field(
        description="Bounding box [x1, y1, x2, y2] in pixels of the upright image (after EXIF rotation)"
    )


class ParticleClassification(BaseModel):
    particle_name: str = Field(description="Name of the detected particle")
    count: int = Field(description="Total count of this particle in the image")
    mean_confidence: float = Field(description="Mean confidence score across all detections of this particle")


class InferenceResult(BaseModel):
    model_version: str = Field(description="Model version from MODEL_VERSION env var")
    particles: dict[str, int] = Field(description="Particle name mapped to count")
    confidence_scores: dict[str, float] = Field(description="Particle name mapped to mean confidence")
    detections: list[Detection] = Field(
        description="Every counted box, after per-class threshold filtering; one per unit in particles"
    )
    raw_detection_count: int = Field(description="Total detections before confidence filtering")
    filtered_detection_count: int = Field(description="Total detections after confidence filtering")
    inference_time_ms: float = Field(description="Time taken to run inference in milliseconds")
    image_width: int = Field(description="Width of the upright image (after EXIF rotation) in pixels")
    image_height: int = Field(description="Height of the upright image (after EXIF rotation) in pixels")