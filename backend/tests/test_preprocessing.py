import numpy as np
from app.preprocessing.clahe_processor import CLAHEProcessor

def test_clahe_preprocessing():
    processor = CLAHEProcessor(clip_limit=2.0, tile_grid_size=8)
    # Generate random test image
    test_image = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    
    result = processor.preprocess(test_image)
    
    assert "original" in result
    assert "denoised" in result
    assert "clahe_enhanced" in result
    assert "normalized" in result
    assert "resized_224" in result
    assert "resized_512" in result
    
    assert result["resized_224"].shape[:2] == (224, 224)
    assert result["resized_512"].shape[:2] == (512, 512)
    
    # Check normalization
    assert result["normalized"].max() <= 1.0
    assert result["normalized"].min() >= 0.0
