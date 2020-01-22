from PIL import Image, ImageDraw, ImageFont

# globals
n = 54
height = 300
width = 500
tag_size = height
path = "./tags/"

for block_id in range(n):
    for j in range(2):
        tag_id = 2 * block_id + j
        tag_name = "tag36_11_" + str(tag_id).zfill(5) + ".png"
        tag_img = Image.open('../resources/tag36h11/' + tag_name)

        # resize the tag
        tag_img = tag_img.resize((tag_size, tag_size), Image.NONE)

        # create empty image with the white background
        res = Image.new('RGB', (width, height), (255, 255, 255))

        # add number to the image
        d = ImageDraw.Draw(res)
        font_size = int(tag_size / 5)
        fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', font_size)
        d.text((10, 10), str(block_id), font=fnt, fill=(0, 0, 0, 255))

        # paste the tag in the center of the image
        res.paste(tag_img, ((width - tag_size) // 2, (height - tag_size) // 2))

        # save image
        if path != None:
            res.save(path + f"texture_block{str(block_id).zfill(2)}_{'a' if j == 0 else 'b'}.png")
            # res.show()