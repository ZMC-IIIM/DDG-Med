import numpy as np
import random
from scipy.special import comb
from PIL import Image
import matplotlib.pyplot as plt
import os


class LocationScaleAugmentation(object):
    def __init__(self, vrange=(0.,1.),  nPoints=4, nTimes=100000):
        self.nPoints=nPoints
        self.nTimes=nTimes
        self.vrange=vrange
        self._get_polynomial_array()

    def _get_polynomial_array(self):
        def bernstein_poly(i, n, t):
            return comb(n, i) * (t ** (n - i)) * (1 - t) ** i
        t = np.linspace(0.0, 1.0, self.nTimes)
        self.polynomial_array = np.array([bernstein_poly(i, self.nPoints - 1, t) for i in range(0, self.nPoints)]).astype(np.float32)

    def get_bezier_curve(self,points):
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        xvals = np.dot(xPoints, self.polynomial_array)
        yvals = np.dot(yPoints, self.polynomial_array)
        return xvals, yvals

    def non_linear_transformation(self, inputs, inverse=False, inverse_prop=0.5):
        start_point, end_point = inputs.min(), inputs.max()
        xPoints = [start_point, end_point]
        yPoints = [start_point, end_point]
        for _ in range(self.nPoints-2):
            xPoints.insert(1, random.uniform(xPoints[0], xPoints[-1]))
            yPoints.insert(1, random.uniform(yPoints[0], yPoints[-1]))
        xvals, yvals = self.get_bezier_curve([[x, y] for x, y in zip(xPoints, yPoints)])
        if inverse and random.random()<=inverse_prop:
            xvals = np.sort(xvals)
        else:
            xvals, yvals = np.sort(xvals), np.sort(yvals)
        return np.interp(inputs, xvals, yvals)

    def location_scale_transformation(self, inputs, slide_limit=20):
        scale = np.array(max(min(random.gauss(1, 0.1), 1.1), 0.9), dtype=np.float32)
        location = np.array(random.gauss(0, 0.5), dtype=np.float32)
        location = np.clip(location, self.vrange[0] - np.percentile(inputs, slide_limit), self.vrange[1] - np.percentile(inputs, 100 - slide_limit))
        return np.clip(inputs*scale + location, self.vrange[0], self.vrange[1])

    def Global_Location_Scale_Augmentation(self, image):
        image = self.non_linear_transformation(image, inverse=False)
        image = self.location_scale_transformation(image).astype(np.float32)
        return image


id_path = []
base_dir = '../../../DG/DoFE/dataset/fundus'

domain_name = "Domain4"


with open(os.path.join(base_dir + "/{}_train.list".format(domain_name)),
            'r') as f:
    id_path = id_path + f.readlines()


with open(os.path.join(base_dir + "/{}_test.list".format(domain_name)), 'r') as f:
    id_path = id_path + f.readlines()

id_path = [item.replace('\n', '') for item in id_path]

print("total {} samples".format(len(id_path)))

location_scale = LocationScaleAugmentation(vrange=(0., 1.))

for index in range(len(id_path)):
    img = Image.open(os.path.join(base_dir, id_path[index].split(' ')[0]))

    id = id_path[index].split(' ')[0]
    id = id.split('/')[-1]
    print(id)
    img = np.array(img).astype(np.float32)
    img /= 127.5
    img -= 1.0
    GLA = location_scale.Global_Location_Scale_Augmentation(img.copy())
    im = Image.fromarray((GLA * 255).astype(np.uint8))
    im.save(os.path.join('../bezier/fundus/Domain4', '{}'.format(id)))



# img1 = Image.open('gdrishtiGS_031.png')
# # 显示图片
# # img.show()
# img = np.array(img1).astype(np.float32)
#
#
# img /= 127.5
# img -= 1.0
#
# location_scale = LocationScaleAugmentation(vrange=(0., 1.))
# GLA = location_scale.Global_Location_Scale_Augmentation(img.copy())
#
#
# # 将NumPy数组转换为Image对象
# im = Image.fromarray((GLA*255).astype(np.uint8))
# im.save(os.path.join('', '111.png'))



