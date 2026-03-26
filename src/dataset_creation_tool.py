# This tool is to create Where's Waldo training samples
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image, ImageDraw
import random
import os

def draw_box(image, box, labels=None, scores=None, color="red", width=2):
    # takes a PIL image and draws bounding boxes on it
    # used for visualizing what the model predicted
    draw = ImageDraw.Draw(image)

    x_min, y_min, x_max, y_max = box
    draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=width)

    text = ""
    if labels is not None:
        text += f"{labels[i]}"
    if scores is not None:
        text += f": {scores[i]:.2f}"

    if text:
        draw.text((x_min, y_min), text, fill=color)

    return image

def resize_image(image: Image, scaling_factor: int):
    w, h = image.size
    new_w = int(w * scaling_factor)
    new_h = int(h * (new_w / w))
    out = image.resize((new_w, new_h), Image.BICUBIC)

    return out

def normalize_image_ratios(background_img, foreground_img, target_ratio = 0.1):
    bg_w, bg_h = background_img.size
    fg_w, fg_h = foreground_img.size
    
    # Calculate scaling based on the larger dimension to ensure it fits
    if fg_w / bg_w > fg_h / bg_h:
        # Scale based on Width
        new_w = int(bg_w * target_ratio)
        new_h = int(new_w * (fg_h / fg_w))
    else:
        # Scale based on Height
        new_h = int(bg_h * target_ratio)
        new_w = int(new_h * (fg_w / fg_h))
        
    return foreground_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

"""
Generates a VOC XML annotation file.

folder_name: Folder name string
filename: Image filename
file_path: Full absolute path to the image
img_width: Image width (px)
img_height: Image height (px)
objects: List of dicts, e.g., [{'name': 'waldo', 'xmin': 706, ...}]
output_path: Path where the .xml will be saved
"""
def bbox_to_xml(folder_name, file_name, file_path, img_width, img_height, objects, output_path):
    # Create the root element
    annotation = ET.Element("annotation")

    # Image Info
    ET.SubElement(annotation, "folder").text = folder_name
    ET.SubElement(annotation, "filename").text = file_name
    ET.SubElement(annotation, "path").text = file_path

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(img_width)
    ET.SubElement(size, "height").text = str(img_height)
    ET.SubElement(size, "depth").text = "3"

    ET.SubElement(annotation, "segmented").text = "0"

    # Add each object
    for obj in objects:
        obj_elem = ET.SubElement(annotation, "object")
        ET.SubElement(obj_elem, "name").text = obj['name']
        ET.SubElement(obj_elem, "pose").text = "Unspecified"
        ET.SubElement(obj_elem, "truncated").text = "0"
        ET.SubElement(obj_elem, "difficult").text = "0"
        
        bndbox = ET.SubElement(obj_elem, "bndbox")
        ET.SubElement(bndbox, "xmin").text = str(obj['xmin'])
        ET.SubElement(bndbox, "ymin").text = str(obj['ymin'])
        ET.SubElement(bndbox, "xmax").text = str(obj['xmax'])
        ET.SubElement(bndbox, "ymax").text = str(obj['ymax'])

    # Format the XML to be pretty (indented)
    xml_str = ET.tostring(annotation, encoding='utf-8')
    xml = minidom.parseString(xml_str).toprettyxml(indent="    ")
    
    # Remove the default <?xml version="1.0" ?> header to match your format
    # (Optional: delete the next line if you want the header included)
    xml = "\n".join(xml.split("\n")[1:])

    with open(os.path.join(output_path, f"{file_name}.xml"), "w") as f:
        f.write(xml)

def parse_xml_bbox_coords(xml_path):
    # Parse the XML data
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Find the bounding box coordinates
    bbox = root.find('object/bndbox')
    xmin = int(bbox.find('xmin').text)
    ymin = int(bbox.find('ymin').text)
    xmax = int(bbox.find('xmax').text)
    ymax = int(bbox.find('ymax').text)

    return xmin, ymin, xmax, ymax

def generate_waldo_data(
    count: int, 
    height: int, 
    width: int, 
    bg_img_path: str, 
    waldo_img_path: str, 
    sample_label_path: str, 
    sample_image_path: str):

    # Count how many files are in the background image directory.
    bg_dir_files = os.listdir(bg_img_path)
    bg_count = 0

    for entry in bg_dir_files:
        if os.path.isfile(os.path.join(bg_img_path, entry)):
            bg_count +=  1

    # Count how many files are in the Waldo image directory.
    waldo_dir_files = os.listdir(waldo_img_path)
    waldo_count = 0;
    
    for entry in waldo_dir_files:
        if os.path.isfile(os.path.join(waldo_img_path, entry)):
            waldo_count +=  1
    
    print(f"There are {bg_count} files in the background image folder.")
    print(f"There are {waldo_count} files in the Waldo image folder.")

    for i in range(count):
        # Randomly pick a background and a waldo image for superimposition.
        bg_file_idx = random.randint(0, bg_count - 1)
        bg = Image.open(os.path.join(bg_img_path, bg_dir_files[bg_file_idx]))
        bg_w, bg_h = bg.size

        # Pick a random Waldo overlay
        waldo_file_idx = random.randint(0, waldo_count - 1)
        waldo = Image.open(os.path.join(waldo_img_path, waldo_dir_files[waldo_file_idx]))

        # Normalize the size of the waldo image to that of the background
        waldo = normalize_image_ratios(bg, waldo)
        scaling_factor = random.uniform(0.7, 1.3)
        waldo = resize_image (waldo, scaling_factor)
        waldo_w, waldo_h = waldo.size
        
        waldo_x = random.randint(0, bg_w - waldo_w)
        waldo_y = random.randint(0, bg_h - waldo_h)
        bg.paste(waldo, (waldo_x, waldo_y), mask = waldo)
        
        sample_w, sample_h = bg.size
# 1. Parse the XML data
        file_name = f"{i}.jpg"
        file_path = os.path.join(sample_image_path, file_name)
        bg.save(file_path, 'JPEG')

        bbox_coords = [{'name': 'waldo', 'xmin': waldo_x, 'ymin': waldo_y, 'xmax': waldo_x + waldo_w, 'ymax': waldo_y + waldo_h}]
        bbox_to_xml(folder_name = "images", file_name = f"{i}", file_path = file_path, img_width = sample_w, img_height = sample_h, objects = bbox_coords, output_path = sample_label_path)

if __name__ == "__main__":
    bg_path = "../data/data_creation/backgrounds"
    waldo_path = "../data/data_creation/waldos"
    sample_save_path = "../data/data_creation/dataset/images"
    label_save_path = "../data/data_creation/dataset/labels"
    
    generate_waldo_data(3, 256, 256, bg_path, waldo_path, label_save_path, sample_save_path)
    sample = Image.open(os.path.join(sample_save_path, "0.jpg"))

    bbox_coords = parse_xml_bbox_coords(os.path.join(label_save_path, "0.xml"))
    bbox_img = draw_box(sample, bbox_coords, labels=None, scores=None, color="green", width=4)
    bbox_img.show()
