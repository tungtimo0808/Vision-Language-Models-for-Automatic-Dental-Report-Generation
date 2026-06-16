| run_name | architecture | depth | layers | test_precision | test_recall | test_f1 | macro_f1 | weighted_f1 | map_50 | map_50_95 | runtime_sec | status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| mbv3_320_shallow | fasterrcnn_mobilenet_v3_large_320_fpn | shallow | 1 | 0.576262 | 0.713687 | 0.637654 | 0.409385 | 0.629115 | 0.302016 | 0.127795 | 3747.68 | ok |
| mbv3_320_deep | fasterrcnn_mobilenet_v3_large_320_fpn | deep | 3 | 0.702536 | 0.793296 | 0.745162 | 0.444762 | 0.742710 | 0.359990 | 0.184170 | 3499.21 | ok |
| mbv3_320_full | fasterrcnn_mobilenet_v3_large_320_fpn | full | 5 | 0.746778 | 0.809358 | 0.776810 | 0.505650 | 0.770690 | 0.409854 | 0.187329 | 4000.05 | ok |
| mbv3_large_shallow | fasterrcnn_mobilenet_v3_large_fpn | shallow | 1 | 0.543798 | 0.697975 | 0.611315 | 0.395763 | 0.601941 | 0.286977 | 0.112834 | 4071.99 | ok |
| mbv3_large_deep | fasterrcnn_mobilenet_v3_large_fpn | deep | 3 | 0.698417 | 0.816690 | 0.752937 | 0.447019 | 0.745775 | 0.370348 | 0.171230 | 3657.45 | ok |
| mbv3_large_full | fasterrcnn_mobilenet_v3_large_fpn | full | 5 | 0.698389 | 0.832751 | 0.759675 | 0.508714 | 0.756634 | 0.426001 | 0.197448 | 4436.24 | ok |
| resnet50_shallow | fasterrcnn_resnet50_fpn | shallow | 1 | 0.772054 | 0.910615 | 0.835630 | 0.589150 | 0.827581 | 0.565010 | 0.293683 | 5044.56 | ok |
| resnet50_deep | fasterrcnn_resnet50_fpn | deep | 3 | 0.815461 | 0.913408 | 0.861660 | 0.595536 | 0.859265 | 0.573890 | 0.313025 | 5220.28 | ok |
| resnet50_full | fasterrcnn_resnet50_fpn | full | 5 | 0.808733 | 0.918296 | 0.860039 | 0.601048 | 0.864049 | 0.595746 | 0.301952 | 3766.44 | ok |
