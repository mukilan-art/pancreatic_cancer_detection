from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from io import BytesIO
from reportlab.pdfgen import canvas
from app.schemas.analysis import ReportRequest
from app.core.config import Settings, get_settings
import os
from pathlib import Path

router = APIRouter()

@router.post("/report")
async def generate_report(
    request: ReportRequest,
    settings: Settings = Depends(get_settings)
):
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.drawString(100, 800, f"Analysis Report for ID: {request.analysis_id}")
        c.drawString(100, 780, f"Generated: {os.path.basename(settings.OUTPUT_DIR)}")
        
        output_dir = Path(settings.OUTPUT_DIR) / request.analysis_id
        if request.include_images and output_dir.exists():
            c.drawString(100, 750, "Includes images.")
            # Basic placement of image if exists
            mask_path = output_dir / "mask.png"
            if mask_path.exists():
                c.drawImage(str(mask_path), 100, 500, width=200, height=200)
                
        c.showPage()
        c.save()
        buffer.seek(0)
        
        return StreamingResponse(
            buffer, 
            media_type="application/pdf", 
            headers={"Content-Disposition": f"attachment; filename=report_{request.analysis_id}.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"detail": str(e), "error_code": "REPORT_ERROR"})
