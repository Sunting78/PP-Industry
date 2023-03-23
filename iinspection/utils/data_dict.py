def post_process_image_info(result):
    for img_path, img_info in result.items():
        preds = img_info['pred']
        img_info['isNG']  = 0
        for pred in preds:
            if pred['isNG'] == 1:
                img_info['isNG'] = 1
                break