from deep_nearest_neighbor.single_image_dataset import image_to_dataset, ImageDataset

def test_image_to_dataset() :
    dataset = ImageDataset(filename="images/mountains.jpg")
    data = next(iter(dataset))
    print('data', data)
    