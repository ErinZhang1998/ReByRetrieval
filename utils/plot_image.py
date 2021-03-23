import matplotlib.pyplot as plt 


def show_image_with_center(img, center):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(111)
    ax1.imshow(img)
    ax1.scatter(center[0], center[1],   marker=".", c='b', s=30)
    plt.show()
    plt.close()