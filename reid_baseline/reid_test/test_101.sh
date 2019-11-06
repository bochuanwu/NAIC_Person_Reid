python3 tools/test.py --config_file='configs/softmax_triplet_with_center_101.yml' MODEL.DEVICE_ID "('0')" DATASETS.NAMES "('market1501')" DATASETS.ROOT_DIR "('../../../reid-mgn/reid-mgn-master/mgn/first_test')" TEST.NECK_FEAT "('after')" TEST.FEAT_NORM "('yes')" MODEL.PRETRAIN_CHOICE "('self')" TEST.WEIGHT "('./market1501/Experiment-all-tricks/resnet101_model_400.pth')"

