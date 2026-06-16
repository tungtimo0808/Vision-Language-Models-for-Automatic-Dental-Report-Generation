| run_name | architecture | depth | layers | min_size | max_size | test_precision | test_recall | test_f1 | test_map_50 | test_map_50_95 | runtime_sec | status | output_dir |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| mbv3_320_shallow | fasterrcnn_mobilenet_v3_large_320_fpn | shallow | 1 | 512 | 768 | 0.625784 | 0.558587 | 0.590279 | 0.486407 | 0.173238 | 1013.55 | ok | faster_rcnn_fdi/outputs/bench/mbv3_320_shallow |
| mbv3_320_deep | fasterrcnn_mobilenet_v3_large_320_fpn | deep | 3 | 512 | 768 | 0.732057 | 0.749213 | 0.740536 | 0.697416 | 0.300888 | 1040.01 | ok | faster_rcnn_fdi/outputs/bench/mbv3_320_deep |
| mbv3_320_full | fasterrcnn_mobilenet_v3_large_320_fpn | full | 5 | 512 | 768 | 0.724758 | 0.760756 | 0.742321 | 0.706711 | 0.302459 | 1331.06 | ok | faster_rcnn_fdi/outputs/bench/mbv3_320_full |
| mbv3_large_shallow | fasterrcnn_mobilenet_v3_large_fpn | shallow | 1 | 512 | 768 | 0.515305 | 0.571179 | 0.541805 | 0.455794 | 0.153631 | 1073.01 | ok | faster_rcnn_fdi/outputs/bench/mbv3_large_shallow |
| mbv3_large_deep | fasterrcnn_mobilenet_v3_large_fpn | deep | 3 | 512 | 768 | 0.655468 | 0.696048 | 0.675148 | 0.637625 | 0.252484 | 1105.66 | ok | faster_rcnn_fdi/outputs/bench/mbv3_large_deep |
| mbv3_large_full | fasterrcnn_mobilenet_v3_large_fpn | full | 5 | 512 | 768 | 0.675685 | 0.655124 | 0.665246 | 0.608746 | 0.246887 | 1201.38 | ok | faster_rcnn_fdi/outputs/bench/mbv3_large_full |
| resnet50_shallow | fasterrcnn_resnet50_fpn | shallow | 1 | 512 | 768 | 0.777653 | 0.966422 | 0.861822 | 0.928387 | 0.513935 | 2303.12 | ok | faster_rcnn_fdi/outputs/bench/resnet50_shallow |
| resnet50_deep | fasterrcnn_resnet50_fpn | deep | 3 | 512 | 768 | 0.818344 | 0.964323 | 0.885356 | 0.927377 | 0.544481 | 2409.73 | ok | faster_rcnn_fdi/outputs/bench/resnet50_deep |
| resnet50_full | fasterrcnn_resnet50_fpn | full | 5 | 512 | 768 | 0.798099 | 0.969220 | 0.875375 | 0.935691 | 0.537745 | 1908.46 | ok | faster_rcnn_fdi/outputs/bench/resnet50_full |
