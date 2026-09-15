"""Both supplied soil orientations and the actual nine-plot MuMu field."""
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from hayday.adb import Screenshot
from hayday.wheating_crop import WheatCropWorker, WheatFarmingVision
from hayday.wheating_field_group import lattice_pitch, observe_group
from hayday.wheating_vision import WheatingVision

FIXTURES = Path(__file__).parent/'fixtures'
ASSETS = Path(__file__).parents[1]/'src/hayday/assets/wheating'


def frame(name):
    return Screenshot((FIXTURES/name).read_bytes(), 1920, 1080, name)


def test_both_rotations_are_found_in_one_image():
    canvas = Image.new('RGB', (1920, 1080), (70, 140, 30))
    centers = []
    for number, position in [(1, (550, 420)), (2, (1020, 530))]:
        with Image.open(ASSETS/f'soil_orientation_{number}.png') as source:
            canvas.paste(source, position, source)
            centers.append((position[0]+source.width//2, position[1]+source.height//2))
    output = io.BytesIO()
    canvas.save(output, 'PNG')
    found = WheatingVision().plots(Screenshot(output.getvalue(), 1920, 1080, 'both'), 'empty')
    assert all(any(np.linalg.norm(np.subtract(target.center, center)) < 18 for target in found)
               for center in centers)


def test_single_page_picker_recognized_without_inventing_page_buttons():
    vision = WheatFarmingVision()
    picker = frame('mumu_soil_picker.png')
    assert vision.seed_menu(picker.png)
    assert vision.page_next(picker.png) is None
    assert vision.empty_plot(picker.png) is not None
    assert vision.full_outline


def test_all_nine_mixed_orientation_plots_are_in_planting_grid():
    vision = WheatFarmingVision()
    picker = frame('mumu_soil_picker.png')
    plot = vision.empty_plot(picker.png)
    points = vision.empty_tiles(picker, plot, 98)
    assert len(points) == len(set(points)) == 9
    expected = [(1162, 569), (1110, 543), (1214, 543), (1057, 569), (1162, 517),
                (1267, 517), (1214, 491), (1320, 491), (1267, 465)]
    assert all(any(np.linalg.norm(np.subtract(p, e)) <= 3 for p in points) for e in expected)


def test_soil_alone_cannot_authorize_planting():
    vision = WheatFarmingVision()
    bare = frame('mumu_soil_bare.png')
    assert not vision.seed_menu(bare.png)
    assert vision.empty_plot(bare.png) is None


def test_group_expands_across_both_soil_orientations():
    vision = WheatFarmingVision()
    picker, bare = frame('mumu_soil_picker.png'), frame('mumu_soil_bare.png')
    plot = vision.empty_plot(picker.png)
    points = vision.empty_tiles(picker, plot, 98)
    moved = WheatCropWorker._translated_plot(picker, bare, points[0], require_visible=False)
    assert moved is not None
    seeds = np.asarray(points[:3])+np.subtract(moved, points[0])
    group = observe_group(bare, seeds, lattice_pitch(points), vision._soil_textures)
    assert group is not None and group.count('bare') == 9


def test_single_seed_arrow_is_not_a_confirmed_crop_fan():
    picker = frame('mumu_soil_picker.png')
    image = Image.open(io.BytesIO(picker.png)).convert('RGB')
    # Remove the corn and soybean arrow tips; wheat alone is insufficient.
    draw = ImageDraw.Draw(image)
    draw.rectangle((925, 340, 1020, 425), fill=(70, 140, 30))
    draw.rectangle((765, 485, 855, 575), fill=(70, 140, 30))
    output = io.BytesIO()
    image.save(output, 'PNG')
    assert not WheatFarmingVision().seed_menu(output.getvalue())
