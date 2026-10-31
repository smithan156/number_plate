from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class Detection(Base):
    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    vehicle_type: Mapped[str] = mapped_column(String(40), index=True)
    vehicle_confidence: Mapped[float] = mapped_column(Float)
    plate_text: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)
    plate_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(40), index=True)
    source: Mapped[str] = mapped_column(String(20), default="image")
    image_path: Mapped[str] = mapped_column(Text)
    annotated_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    plate_image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    vehicle_box: Mapped[str] = mapped_column(Text)
    plate_box: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
