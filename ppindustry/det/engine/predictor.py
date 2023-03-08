import os
from tqdm import tqdm
import typing

from ppdet.core.workspace import create
from ppdet.engine import Trainer
from ppdet.data.source.category import get_categories
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.metrics import get_infer_results
import numpy as np
from PIL import Image, ImageOps, ImageFile
from ppindustry.utils.logger import setup_logger
logger = setup_logger('Predictor')

class Predictor(Trainer):

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
                save_results=False,
                visualize=True):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        anno_file = self.dataset.get_anno()
        clsid2catid, catid2name = get_categories(
            self.cfg.metric, anno_file=anno_file)

        # Run Infer 
        self.status['mode'] = 'test'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('TestReader')(self.dataset, 0)
            self._flops(flops_loader)
        results = []
        

        for step_id, data in enumerate(tqdm(loader)):
            self.status['step_id'] = step_id
            # forward
            outs = self.model(data)
            
            for key in ['im_shape', 'scale_factor', 'im_id']:
                if isinstance(data, typing.Sequence):
                    outs[key] = data[0][key]
                else:
                    outs[key] = data[key]
            for key, value in outs.items():
                if hasattr(value, 'numpy'):
                    outs[key] = value.numpy()
            #outs['im_path'] = imid2path[int(outs['im_id'])]
            infer_res = self.get_det_res(
                outs['bbox'], outs['bbox_num'], outs['im_id'], clsid2catid, imid2path)

            results.append(infer_res)

        if visualize:
            for outs in results:
                batch_res = get_infer_results(outs, clsid2catid)
                bbox_num = outs['bbox_num']
                start = 0
                for i, im_id in enumerate(outs['im_id']):
                    image_path = imid2path[int(im_id)]
                    image = Image.open(image_path).convert('RGB')
                    image = ImageOps.exif_transpose(image)
                    self.status['original_image'] = np.array(image.copy())

                    end = start + bbox_num[i]
                    bbox_res = batch_res['bbox'][start:end] \
                            if 'bbox' in batch_res else None

                for outs in results:
                    batch_res = get_infer_results(outs, clsid2catid)
                    bbox_num = outs['bbox_num']
                    
                    start = 0
                    for i, im_id in enumerate(outs['im_id']):
                        image_path = imid2path[int(im_id)]
                        image = Image.open(image_path).convert('RGB')
                        image = ImageOps.exif_transpose(image)
                        self.status['original_image'] = np.array(image.copy())

                        end = start + bbox_num[i]
                        bbox_res = batch_res['bbox'][start:end] \
                                if 'bbox' in batch_res else None
                        mask_res = batch_res['mask'][start:end] \
                                if 'mask' in batch_res else None
                        segm_res = batch_res['segm'][start:end] \
                                if 'segm' in batch_res else None
                        keypoint_res = batch_res['keypoint'][start:end] \
                                if 'keypoint' in batch_res else None
                        pose3d_res = batch_res['pose3d'][start:end] \
                                if 'pose3d' in batch_res else None
                        image = visualize_results(
                            image, bbox_res, mask_res, segm_res, keypoint_res,
                            pose3d_res, int(im_id), catid2name, draw_threshold)
                        self.status['result_image'] = np.array(image.copy())
                        if self._compose_callback:
                            self._compose_callback.on_step_end(self.status)
                        # save image with detection
                        save_name = self._get_save_image_name(output_dir,
                                                            image_path)
                        logger.info("Detection bbox results save in {}".format(
                            save_name))
                        image.save(save_name, quality=95)

                        start = end

        return results

    def get_det_res(self, bboxes, bbox_nums, image_id, label_to_cat_id_map, imid2path, bias=0):
        det_res = []
        k = 0
        for i in range(len(bbox_nums)):
            cur_image_id = int(image_id[i][0])
            cur_image_path = imid2path[cur_image_id]
            det_nums = bbox_nums[i]
            for j in range(det_nums):
                dt = bboxes[k]
                k = k + 1
                num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
                if int(num_id) < 0:
                    continue
                category_id = label_to_cat_id_map[int(num_id)]
                w = xmax - xmin + bias
                h = ymax - ymin + bias
                bbox = [xmin, ymin, w, h]
                dt_res = {
                    'image_id': cur_image_id,
                    'image_path': cur_image_path,
                    'category_id': category_id,
                    'bbox': bbox,
                    'score': score
                }
                det_res.append(dt_res)
        return det_res