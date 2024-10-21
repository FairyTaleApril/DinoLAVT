"""
This interface provides access to four datasets:
1) refclef
2) refcoco
3) refcoco+
4) refcocog
split by unc and google

The following API functions are defined:
REFER      - REFER api class
getRefIds  - get ref ids that satisfy given filter conditions.
getAnnIds  - get ann ids that satisfy given filter conditions.
getImgIds  - get image ids that satisfy given filter conditions.
getCatIds  - get category ids that satisfy given filter conditions.
loadRefs   - load refs with the specified ref ids.
loadAnns   - load anns with the specified ann ids.
loadImgs   - load images with the specified image ids.
loadCats   - load category names with the specified category ids.
getRefBox  - get ref's bounding box [x, y, w, h] given the ref_id
showRef    - show image, segmentation or box of the referred object with the ref
getMask    - get mask and area of the referred object given ref
showMask   - show mask of the referred object given ref
"""

import itertools
import json
import os.path as osp
import pickle as pickle
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from pycocotools import mask

from utils.args_parser import get_args_parser


class REFER:
    def __init__(self, ref_root, img_dir, dataset='refcoco+', splitBy='unc', max_image_num=None):
        # Provide data_root folder which contains refclef, refcoco, refcoco+ and refcocog,
        # also provide dataset name and splitBy information
        # e.g., dataset = 'refcoco', splitBy = 'unc'
        print('Loading dataset %s into memory...' % dataset)
        self.DATA_DIR = osp.join(ref_root, dataset)
        self.IMAGE_DIR = img_dir
        self.data = {}
        self.kept_image_ids = []
        self.kept_ref = []

        self.Refs = {}
        self.Anns = {}
        self.Imgs = {}
        self.Cats = {}
        self.Sents = {}
        self.imgToRefs = {}
        self.imgToAnns = {}
        self.refToAnn = {}
        self.annToRef = {}
        self.catToRefs = {}
        self.sentToRef = {}
        self.sentToTokens = {}

        tic = time.time()

        # Load refs from data/dataset/refs(splitBy).p
        ref_file_path = osp.join(self.DATA_DIR, 'refs(' + splitBy + ').p')
        refs = pickle.load(open(ref_file_path, 'rb'))
        self.data['dataset'] = dataset
        self.data['refs'] = refs

        # Load annotations from data/dataset/instances.json
        instances_path = osp.join(self.DATA_DIR, 'instances.json')
        instances = json.load(open(instances_path, 'r'))
        self.data['images'] = instances['images']
        self.data['annotations'] = instances['annotations']
        self.data['categories'] = instances['categories']

        # Create index
        if max_image_num is not None:
            max_image_num = min(max_image_num, len(self.data['refs']))
        self.createIndex(max_image_num)
        print('Data loaded (t = %.2fs)' % (time.time() - tic))

    def createIndex(self, max_image_num):
        # Create sets of mapping
        # 1)  Refs: 	 	{ref_id: ref}
        # 2)  Anns: 	 	{ann_id: ann}
        # 3)  Imgs:		 	{image_id: image}
        # 4)  Cats: 	 	{category_id: category_name}
        # 5)  Sents:     	{sent_id: sent}
        # 6)  imgToRefs: 	{image_id: refs}
        # 7)  imgToAnns: 	{image_id: anns}
        # 8)  refToAnn:  	{ref_id: ann}
        # 9)  annToRef:  	{ann_id: ref}
        # 10) catToRefs: 	{category_id: refs}
        # 11) sentToRef: 	{sent_id: ref}
        # 12) sentToTokens: {sent_id: tokens}
        print('Creating index...')

        for ref in self.data['refs']:
            image_id = ref['image_id']
            if image_id not in self.kept_image_ids:
                if len(self.kept_image_ids) == max_image_num:
                    break
                self.kept_image_ids.append(image_id)

        for ref in self.data['refs']:
            if ref['image_id'] in self.kept_image_ids:
                self.kept_ref.append(ref)

        # Fetch info from refs
        for ref in self.kept_ref:
            self.Refs[ref['ref_id']] = ref
            self.imgToRefs[ref['image_id']] = self.imgToRefs.get(ref['image_id'], []) + [ref]
            self.catToRefs[ref['category_id']] = self.catToRefs.get(ref['category_id'], []) + [ref]
            self.annToRef[ref['ann_id']] = ref

            for sent in ref['sentences']:
                self.Sents[sent['sent_id']] = sent
                self.sentToRef[sent['sent_id']] = ref
                self.sentToTokens[sent['sent_id']] = sent['tokens']

        # Fetch info from instances
        for ann in self.data['annotations']:
            if ann['id'] in self.annToRef:
                self.Anns[ann['id']] = ann
                self.imgToAnns[ann['image_id']] = self.imgToAnns.get(ann['image_id'], []) + [ann]
        for img in self.data['images']:
            if img['id'] in self.imgToRefs:
                self.Imgs[img['id']] = img
        for cat in self.data['categories']:
            if cat['id'] in self.catToRefs:
                self.Cats[cat['id']] = cat['name']

        for ref in self.kept_ref:
            self.refToAnn[ref['ref_id']] = self.Anns[ref['ann_id']]

        print('Index created.')

    def getRefIds(self, image_ids=None, cat_ids=None, ref_ids=None, split=''):
        image_ids = [] if image_ids is None else image_ids
        cat_ids = [] if cat_ids is None else cat_ids
        ref_ids = [] if ref_ids is None else ref_ids

        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.kept_ref
        else:
            if not len(image_ids) == 0:
                refs = [self.imgToRefs[image_id] for image_id in image_ids]
            else:
                refs = self.kept_ref
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref['category_id'] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref['ref_id'] in ref_ids]
            if not len(split) == 0:
                if split in ['testA', 'testB', 'testC']:
                    refs = [ref for ref in refs if split[-1] in ref['split']]  # We also consider testAB, testBC, ...
                elif split in ['testAB', 'testBC', 'testAC']:
                    refs = [ref for ref in refs if ref['split'] == split]  # Rarely used I guess...
                elif split == 'test':
                    refs = [ref for ref in refs if 'test' in ref['split']]
                elif split == 'train' or split == 'val':
                    refs = [ref for ref in refs if ref['split'] == split]
                else:
                    print('No such split [%s]' % split)
                    sys.exit()
        ref_ids = [ref['ref_id'] for ref in refs]
        return ref_ids

    def getImgIds(self, ref_ids=None):
        ref_ids = [] if ref_ids is None else ref_ids
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if not len(ref_ids) == 0:
            image_ids = list(set([self.Refs[ref_id]['image_id'] for ref_id in ref_ids]))
        else:
            image_ids = self.Imgs.keys()
        return image_ids

    def getCatIds(self):
        return self.Cats.keys()

    def loadRefs(self, ref_ids=None):
        ref_ids = [] if ref_ids is None else ref_ids

        if type(ref_ids) == list:
            return [self.Refs[ref_id] for ref_id in ref_ids]
        elif type(ref_ids) == int:
            return [self.Refs[ref_ids]]

    def loadAnns(self, ann_ids=None):
        ann_ids = [] if ann_ids is None else ann_ids

        if type(ann_ids) == list:
            return [self.Anns[ann_id] for ann_id in ann_ids]
        elif type(ann_ids) == int or type(ann_ids) == str:
            return [self.Anns[ann_ids]]

    def loadImgs(self, image_ids=None):
        image_ids = [] if image_ids is None else image_ids

        if type(image_ids) == list:
            return [self.Imgs[image_id] for image_id in image_ids]
        elif type(image_ids) == int:
            return [self.Imgs[image_ids]]

    def loadCats(self, cat_ids=None):
        cat_ids = [] if cat_ids is None else cat_ids

        if type(cat_ids) == list:
            return [self.Cats[cat_id] for cat_id in cat_ids]
        elif type(cat_ids) == int:
            return [self.Cats[cat_ids]]

    def getRefBox(self, ref_id):
        ann = self.refToAnn[ref_id]
        return ann['bbox']  # [x, y, w, h]

    def showRef(self, ref, seg_box='seg'):
        ax = plt.gca()
        # Show image
        image = self.Imgs[ref['image_id']]
        I = io.imread(osp.join(self.IMAGE_DIR, image['file_name']))
        ax.imshow(I)

        # Show refer expression
        for sid, sent in enumerate(ref['sentences']):
            print('%s. %s' % (sid + 1, sent['sent']))
        # Show segmentations
        if seg_box == 'seg':
            ann = self.Anns[ref['ann_id']]
            polygons = []
            color = []
            c = 'none'
            if type(ann['segmentation'][0]) == list:
                # Polygon used for refcoco*
                for seg in ann['segmentation']:
                    poly = np.array(seg).reshape((len(seg) // 2, 2))
                    polygons.append(Polygon(poly, closed=True, alpha=0.4))
                    color.append(c)
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 1, 0, 0), linewidths=3, alpha=1)
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(polygons, facecolors=color, edgecolors=(1, 0, 0, 0), linewidths=1, alpha=1)
                ax.add_collection(p)  # thin red polygon
            else:
                # Mask used for refclef
                rle = ann['segmentation']
                m = mask.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # Show bounding-box
        elif seg_box == 'box':
            bbox = self.getRefBox(ref['ref_id'])
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='green', linewidth=3)
            ax.add_patch(box_plot)

    def getMask(self, ref):
        # Return mask, area and mask-center
        ann = self.refToAnn[ref['ref_id']]
        image = self.Imgs[ref['image_id']]

        if type(ann['segmentation'][0]) == list:  # polygon
            rle = mask.frPyObjects(ann['segmentation'], image['height'], image['width'])
        else:
            rle = ann['segmentation']

        m = mask.decode(rle)
        m = np.sum(m, axis=2).astype(np.uint8)  # There are multiple binary maps corresponding to multiple segs
        # Compute area
        area = sum(mask.area(rle))  # Should be close to ann['area']
        return {'mask': m, 'area': area}

    def showMask(self, ref):
        M = self.getMask(ref)
        msk = M['mask']
        ax = plt.gca()
        ax.imshow(msk)


def main():
    args = get_args_parser()
    refer = REFER(ref_root='', img_dir=args.img_dir, splitBy='unc', max_data_size=5)

    ref_ids = refer.getRefIds(split='train')
    print('There are %s training referred objects.' % len(ref_ids))

    for ref_id in ref_ids:
        ref = refer.loadRefs(ref_id)[0]
        print('The label is %s.' % refer.Cats[ref['category_id']])
        plt.figure()
        refer.showRef(ref, seg_box='box')
        refer.showRef(ref, seg_box='seg')
        plt.show()


if __name__ == '__main__':
    main()
