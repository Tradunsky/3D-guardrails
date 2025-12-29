import os
import asyncio
from dddguardrails.rendering import render_tiled_views
from pathlib import Path

async def test_tiled_render():
    os.environ["DDDG_DUMP_SCREENSHOTS"] = "True"
    data_dir = Path("tests/data")
    asset_path = data_dir / "candy.glb"
    
    if not asset_path.exists():
        print(f"Asset not found: {asset_path}")
        return

    contents = asset_path.read_bytes()
    extension = "glb"
    resolution = (1024, 1024)
    
    print("Rendering tiled views...")
    screenshot = render_tiled_views(contents, extension, resolution)
    
    with open("screenshot-tiled-test.png", "wb") as f:
        f.write(screenshot)
    print("Saved screenshot-tiled-test.png")

if __name__ == "__main__":
    asyncio.run(test_tiled_render())
