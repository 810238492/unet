from PIL import Image

def keep_image_size_open(path,size=(512,512)):
    print(path)
    img=Image.open(path)
    temp=max(img.size)
    mask=Image.new('RGB',(temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask=mask.resize(size)
    return mask
