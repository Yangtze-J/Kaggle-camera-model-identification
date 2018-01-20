import os
import re
import wget

ROOT_DIR = os.getcwd()
DEFAULT_WEIGHT_PATH = os.path.join(ROOT_DIR, "model")

with open(ROOT_DIR+'/good_jpgs') as f:
    content = f.readlines()
# you may also want to remove whitespace characters like `\n` at the end of each line
content = [x.strip() for x in content]

for i, name in enumerate(content):
    # Find directory and image name
    try:
        directory = re.search('/(.+?)/', name).group(1)
        start = name.find(directory+'/') + len(directory) + 1
        image_name = name[start:len(name)]
    except AttributeError:
        continue
    # Open directory's urls_final
    with open(ROOT_DIR + '/' + directory + '/urls_final' ) as sf:
        urls = sf.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    urls = [x.strip() for x in urls]

    try:
        # Check if image exist.
        if os.path.isfile(ROOT_DIR + '/' + directory + '/' + image_name) is False:
            matching_url = [s for s in urls if image_name in s]
            wget.download(matching_url[0], ROOT_DIR + '/' + directory + '/' + image_name)
            print("\nFinished downloading image : ", directory, '/', image_name)
        else:
            print("Image", directory, '/', image_name, "has been downloaded")
    except AttributeError:
        print("Can not download image", directory, '/', image_name)

print("Finish all good jpgs downloading")

