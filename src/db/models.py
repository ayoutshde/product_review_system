from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from src.db.database import Base


class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)

    order_code = Column(String(100), index=True, nullable=False)
    camera = Column(String(50), index=True, nullable=False)

    video_name = Column(String(500), nullable=False)
    video_path = Column(Text, nullable=False)

    device_id = Column(String(100), nullable=True)
    session_id = Column(String(100), nullable=True)
    date = Column(String(50), nullable=True)
    time = Column(String(50), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    frames = relationship("Frame", back_populates="video")


class Frame(Base):
    __tablename__ = "frames"

    id = Column(Integer, primary_key=True, index=True)

    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)

    order_code = Column(String(100), index=True, nullable=False)
    camera = Column(String(50), index=True, nullable=False)

    frame_number = Column(Integer, index=True, nullable=False)
    frame_path = Column(Text, nullable=False)

    timestamp = Column(Float, nullable=True)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="frames")
    prediction = relationship("Prediction", back_populates="frame", uselist=False)
    queue_items = relationship("QueueItem", back_populates="frame")
    reviews = relationship("Review", back_populates="frame")
    annotations = relationship("Annotation", back_populates="frame")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)

    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False, unique=True)

    model_name = Column(String(255), nullable=True)
    prediction_status = Column(String(50), index=True, nullable=False)

    max_confidence = Column(Float, default=0.0)
    box_count = Column(Integer, default=0)

    boxes_json = Column(JSONB, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    frame = relationship("Frame", back_populates="prediction")
    queue_items = relationship("QueueItem", back_populates="prediction")
    reviews = relationship("Review", back_populates="prediction")


class QueueItem(Base):
    __tablename__ = "queue_items"

    id = Column(Integer, primary_key=True, index=True)

    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    queue_type = Column(String(100), index=True, nullable=False)
    queue_reason = Column(String(100), index=True, nullable=False)

    decision = Column(String(100), nullable=True)
    warning = Column(String(100), nullable=True)

    status = Column(String(50), index=True, default="pending")

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    frame = relationship("Frame", back_populates="queue_items")
    prediction = relationship("Prediction", back_populates="queue_items")
    reviews = relationship("Review", back_populates="queue_item")
    annotations = relationship("Annotation", back_populates="queue_item")


class Review(Base):
    __tablename__ = "reviews"

    id = Column(Integer, primary_key=True, index=True)

    queue_item_id = Column(Integer, ForeignKey("queue_items.id"), nullable=False)
    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=True)

    review_status = Column(String(100), index=True, nullable=False)

    # Examples:
    # correct_detection
    # wrong_box
    # box_needs_split
    # real_missed_detection
    # correct_no_detection
    # ignore_frame

    note = Column(Text, nullable=True)
    reviewed_by = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    queue_item = relationship("QueueItem", back_populates="reviews")
    frame = relationship("Frame", back_populates="reviews")
    prediction = relationship("Prediction", back_populates="reviews")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True, index=True)

    frame_id = Column(Integer, ForeignKey("frames.id"), nullable=False)
    queue_item_id = Column(Integer, ForeignKey("queue_items.id"), nullable=True)

    annotation_source = Column(String(100), index=True, nullable=False)

    # Examples:
    # model_prediction
    # manual_draw
    # corrected_model_box
    # missed_detection_manual

    label_name = Column(String(100), default="product")
    status = Column(String(50), index=True, default="saved")

    created_by = Column(String(100), nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())

    frame = relationship("Frame", back_populates="annotations")
    queue_item = relationship("QueueItem", back_populates="annotations")
    boxes = relationship("AnnotationBox", back_populates="annotation")


class AnnotationBox(Base):
    __tablename__ = "annotation_boxes"

    id = Column(Integer, primary_key=True, index=True)

    annotation_id = Column(Integer, ForeignKey("annotations.id"), nullable=False)

    class_id = Column(Integer, default=0)
    label_name = Column(String(100), default="product")

    box_type = Column(String(50), default="obb")

    points_json = Column(JSONB, nullable=False)
    normalized_points_json = Column(JSONB, nullable=True)

    confidence = Column(Float, nullable=True)

    box_source = Column(String(100), nullable=False)

    # Examples:
    # model
    # manual
    # corrected

    created_at = Column(DateTime(timezone=True), server_default=func.now())

    annotation = relationship("Annotation", back_populates="boxes")


class CameraRegion(Base):
    __tablename__ = "camera_regions"

    id = Column(Integer, primary_key=True, index=True)

    camera = Column(String(50), index=True, nullable=False)
    region_name = Column(String(100), index=True, nullable=False)

    # Examples:
    # fridge_polygon
    # takeout_zone
    # ignore_zone

    points_json = Column(JSONB, nullable=False)

    created_at = Column(DateTime(timezone=True), server_default=func.now())