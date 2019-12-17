from PIL import Image, ImageDraw, ImageFont
import sys


def generate_texture_with_tag(block_num, side, path):
    assert side in ['left', 'right'], "Wrong attribute side: ".format(side)

    tag_size = 300
    width = 300
    height = 500
    rotation = None
    letter = None

    if side == 'left':
        tag_id = 2 * block_num
        rotation = -90
        letter = "L"
    else:
        tag_id = 2 * block_num + 1
        rotation = 90
        letter = "R"

    tag_name = "tag36_11_" + str(tag_id).zfill(5) + ".png"
    tag_img = Image.open('../../resources/tag36h11/' + tag_name)
    tag_img = tag_img.resize((tag_size, tag_size), Image.NONE)
    tag_w, tag_h = tag_img.size

    # rotate tag
    tag_img = tag_img.rotate(rotation)

    # create empty image with the white background
    res = Image.new('RGB', (width, height), (255, 255, 255))

    # add letter to the texture
    d = ImageDraw.Draw(res)
    fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 50)
    d.text((50, 50), letter, direction="upwards", font=fnt, stroke_width=2, fill=(0, 0, 0, 255))

    # paste the tag in the center of the image
    res.paste(tag_img, ((width - tag_w) // 2, (height - tag_h) // 2))



    # resize the texture
    res = res.resize((max(height, width), max(height, width)), Image.NONE)

    if path != None:
        # res.save(path + f"texture_block{block_num}_{side}.png")
        res.show()
    return res

def generate_texture_for_coordinate_frame():
    tag_id = 257
    tag_name = "tag36_11_" + str(tag_id).zfill(5) + ".png"
    tag_img = Image.open('../../resources/tag36h11/' + tag_name)
    tag_img = tag_img.resize((1000, 1000))
    tag_img.save("../images/coordinate_frame_texture3.png")

def generate_square_texture():
    tag_id = 300
    tag_name = "tag36_11_" + str(tag_id).zfill(5) + ".png"
    tag_img = Image.open('../../resources/tag36h11/' + tag_name)
    # rotate tag
    tag_img = tag_img.rotate(90)
    tag_img = tag_img.resize((1000, 1000))
    tag_img.save("../images/floating_object_texture.png")


if __name__ == "__main__":
     for i in range(1):
         # generate_texture_with_tag(i, 'left', "../images/")
         generate_texture_with_tag(i, 'right', "../images/")

    # generate_texture_for_coordinate_frame()
    # generate_square_texture()



