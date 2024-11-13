import os
from .mevis import register_mevis_instances


# ====    Predefined splits for mevis    ===========
_PREDEFINED_SPLITS_mevis = {
    "mevis_train": ("mevis/train",
                   "mevis/train/meta_expressions.json"),
    "mevis_val": ("mevis/valid_u",
                 "mevis/valid_u/meta_expressions.json"),
    "mevis_test": ("mevis/valid",
                  "mevis/valid/meta_expressions.json"),
    "mevis_challenge": ("mevis/test",
                  "mevis/test/meta_expressions.json"),
}

_PREDEFINED_SPLITS_ytvos = {
    "ytvos_train": ("ref-youtube-vos/train",
                    "ref-youtube-vos/meta_expressions/train/meta_expressions.json"),
    "ytvos_val": ("ref-youtube-vos/valid",
                  "ref-youtube-vos/meta_expressions/valid/meta_expressions.json"),
    "ytvos_test": ("ref-youtube-vos/test",
                   "ref-youtube-vos/meta_expressions/test/meta_expressions.json"),
}

_PREDEFINED_SPLITS_davis = {
    "davis_train": ("ref-davis/train",
                    "ref-davis/meta_expressions/train/meta_expressions.json"),
    "davis_val": ("ref-davis/valid",
                  "ref-davis/meta_expressions/valid/meta_expressions.json"),
}

def register_all_mevis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_mevis.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_mevis_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

def register_all_ytvos(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_ytvos.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvos_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root)
        )

def register_all_davis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_davis.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvos_instances(
            key,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root)
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_mevis(_root)
    register_all_ytvos(_root)
    register_all_davis(_root)



