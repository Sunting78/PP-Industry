from .defect_dataset import *

@COMMON_PIPELINES.register_module()
class CopyAndPasteV2(object):
    def __init__(
        self,
        cats="all",
        paste_prob=0.5,
        crop_prob=1.0,
        random_crop_extend_x=(50, 300),
        random_crop_extend_y=(50, 300),
        resize_range=(0.5, 1.5),
        pos_ratio=0.9,
        neg_ratio=0.3,
        buffer_size=20,
        max_paste_number=3,
        crop_try_number=10,
        paste_try_number=10,
        use_blur=False,
    ):
        # params
        self.paste_prob = paste_prob
        self.crop_prob = crop_prob
        self.max_paste_number = max_paste_number
        self.crop_try_number = crop_try_number
        self.paste_try_number = paste_try_number
        self.cats = cats
        self.buffer_size = buffer_size

        self.pos_ratio = pos_ratio
        self.neg_ratio = neg_ratio

        self.random_crop_extend_x = random_crop_extend_x
        self.random_crop_extend_y = random_crop_extend_y

        self.resize_range = resize_range

        self.use_blur = use_blur

        # buffer
        self.buffers = []

    def get_random_crop_bbox(self, image, target_box):
        h, w, _ = image.shape
        xmin, ymin, xmax, ymax = target_box
        crop_xmin = max(0, xmin - np.random.randint(*self.random_crop_extend_x))
        crop_xmax = min(w, xmax + np.random.randint(*self.random_crop_extend_x))
        crop_ymin = max(0, ymin - np.random.randint(*self.random_crop_extend_y))
        crop_ymax = min(h, ymax + np.random.randint(*self.random_crop_extend_y))
        return np.array([crop_xmin, crop_ymin, crop_xmax, crop_ymax])

    def get_occlusion_bbox(self, crop_bbox, gt_bboxes, pos_ratio, neg_ratio):
        valid_flags = []
        for gt_bbox in gt_bboxes:
            occlusion = float(
                self.box_intersection(gt_bbox, crop_bbox)
            ) / self.box_area(gt_bbox)
            if occlusion <= neg_ratio:
                valid_flags.append(0)
            elif occlusion >= pos_ratio:
                valid_flags.append(1)
            else:
                # occlusion > neg_ratio and occlusion < pos_ratio:
                valid_flags.append(-1)
        return np.array(valid_flags)

    def blur(self, img, kernel_size, is_vertical=False):
        if kernel_size == 0:
            return img
        kernel_v = np.zeros((kernel_size, kernel_size))
        kernel_h = np.copy(kernel_v)
        kernel_v[:, int((kernel_size - 1) / 2)] = np.ones(kernel_size)
        kernel_h[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        # Normalize
        kernel_v /= kernel_size
        kernel_h /= kernel_size
        if is_vertical:
            img = cv2.filter2D(img, -1, kernel_v)
        else:
            img = cv2.filter2D(img, -1, kernel_h)
        return img

    def box_intersection(self, boxa, boxb):
        # ç›¸äº¤çŸ©å½¢
        xa = max(boxa[0], boxb[0])
        ya = max(boxa[1], boxb[1])
        xb = min(boxa[2], boxb[2])
        yb = min(boxa[3], boxb[3])
        inter_area = max(0, xb - xa + 1) * max(0, yb - ya + 1)
        return inter_area

    def box_area(self, box):
        return (box[2] - box[0]) * (box[3] - box[1])

    def create_rectrangle_gaussian_masks(
        self, base_shape=(50, 50, 3), border=5, kernel=(21, 21), sigma_x=0
    ):
        img = np.zeros(base_shape, dtype=np.float32)
        img[border : base_shape[0] - border, border : base_shape[1] - border, :] = 1
        return cv2.GaussianBlur(img, kernel, sigma_x)

    def copy_and_save(self, results):

        if "gt_bboxes" not in results.keys() or not len(results["gt_bboxes"]) > 0:
            return

        gt_bboxes = results["gt_bboxes"]
        gt_labels = results["gt_labels"]
        image = results["img"]

        for idx, target_box in enumerate(gt_bboxes):
            if (
                self.cats == "all"
                or gt_labels[idx] in self.cats
                and np.random.random() <= self.crop_prob
            ):

                for _ in range(self.crop_try_number):
                    crop_bbox = self.get_random_crop_bbox(image, target_box)
                    valid_flags = self.get_occlusion_bbox(
                        crop_bbox, gt_bboxes, self.pos_ratio, self.neg_ratio
                    )
                    if -1 not in valid_flags:
                        break
                if -1 in valid_flags:
                    continue

                crop_x1, crop_y1, crop_x2, crop_y2 = crop_bbox

                dst = image[int(crop_y1) : int(crop_y2), int(crop_x1) : int(crop_x2)]

                save_bboxes = gt_bboxes - np.array([crop_x1, crop_y1, crop_x1, crop_y1])
                save_bboxes[:, 0] = np.clip(save_bboxes[:, 0], 0, None)
                save_bboxes[:, 1] = np.clip(save_bboxes[:, 1], 0, None)
                save_bboxes[:, 2] = np.clip(save_bboxes[:, 2], None, crop_x2 - crop_x1)
                save_bboxes[:, 3] = np.clip(save_bboxes[:, 3], None, crop_y2 - crop_y1)

                save_bboxes = save_bboxes[valid_flags == 1]
                save_gt_labels = gt_labels[valid_flags == 1]

                if len(save_gt_labels) > 0:
                    self.buffers.append((dst, save_bboxes, save_gt_labels))

    def paste_augmentation(self, img, bboxes):
        if self.use_blur:
            blur_type = np.random.random()
            if blur_type < 0.3:
                blur_value = np.random.randint(1, 8)
                img = cv2.blur(img, (blur_value, blur_value))
            elif blur_type < 0.6:
                blur_value = np.random.choice([3, 5, 7])
                img = cv2.GaussianBlur(img, (blur_value, blur_value), 2)
            elif blur_type < 0.7:
                img = self.blur(img, np.random.randint(1, 8))

        # random resize
        resize_ratio = np.random.uniform(*self.resize_range)
        img = cv2.resize(img, (0, 0), fx=resize_ratio, fy=resize_ratio)
        bboxes = np.round(bboxes * resize_ratio)

        return img, bboxes

    def select_and_paste(self, results):
        origin_image = copy.deepcopy(results["img"])
        gt_bboxes = copy.deepcopy(results["gt_bboxes"])
        gt_labels = copy.deepcopy(results["gt_labels"])

        for _ in range(self.paste_try_number):
            if len(self.buffers) == 0:
                break
            select_idx = np.random.choice(range(len(self.buffers)))
            select_img, select_bboxes, select_labels = self.buffers[select_idx]
            select_img, select_bboxes = self.paste_augmentation(
                select_img, select_bboxes
            )
            select_h, select_w, _ = select_img.shape
            origin_h, origin_w, _ = origin_image.shape
            if origin_w - select_w <= 0 or origin_h - select_h <= 0:
                continue
            x1_put = np.random.randint(0, origin_w - select_w)
            y1_put = np.random.randint(0, origin_h - select_h)
            x2_put = x1_put + select_w
            y2_put = y1_put + select_h
            crop_bbox = [x1_put, y1_put, x2_put, y2_put]
            valid_flags = self.get_occlusion_bbox(
                crop_bbox, gt_bboxes, 1 - self.neg_ratio, 1 - self.pos_ratio
            )
            if -1 not in valid_flags:
                mask = self.create_rectrangle_gaussian_masks(
                    base_shape=select_img.shape
                )
                origin_image[y1_put:y2_put, x1_put:x2_put] = (
                    origin_image[y1_put:y2_put, x1_put:x2_put] * (1 - mask)
                    + mask * select_img
                )
                gt_bboxes = gt_bboxes[valid_flags == 0]
                gt_labels = gt_labels[valid_flags == 0]
                select_bboxes += [x1_put, y1_put, x1_put, y1_put]
                gt_bboxes = np.concatenate([gt_bboxes, select_bboxes])
                gt_labels = np.concatenate([gt_labels, select_labels])
                self.buffers.pop(select_idx)
                break

        results["img"] = origin_image
        results["gt_bboxes"] = np.array(gt_bboxes, dtype=np.float32).reshape([-1, 4])
        results["gt_labels"] = np.array(gt_labels, dtype=np.int64)
        return results

    def __call__(self, results):
        new_results = copy.deepcopy(results)

        if (
            np.random.random() <= self.crop_prob
            and len(self.buffers) >= self.buffer_size
        ):
            paste_number = np.random.randint(1, self.max_paste_number)
            for _ in range(paste_number):
                new_results = self.select_and_paste(new_results)

        self.copy_and_save(results)
        for _ in range(len(self.buffers) - self.buffer_size):
            self.buffers.pop(0)

        return new_results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(buffer_size={self.buffer_size}, "
        return repr_str