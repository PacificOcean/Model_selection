#/bin/sh

cd tools_for_cv_2
ln -s ../subscripts/evaluate_by_cv.py evaluate.py
ln -s ../subscripts/ga.py .
ln -s ../subscripts/gene_ga_data.py .
ln -s ../subscripts/get_best_result.sh .
ln -s ../subscripts/ipaexg.ttf .
ln -s ../subscripts/output_thresh_confmat.py .
ln -s ../subscripts/plot_confmat.py .
ln -s ../subscripts/plot_ga.py .
ln -s ../subscripts/plot_score_dist.py .
ln -s ../subscripts/run_ga.py .
ln -s ../subscripts/utils.py .

cd ../tools_for_cv_multi
ln -s ../subscripts/evaluate_by_cv.py evaluate.py
ln -s ../subscripts/ga.py .
ln -s ../subscripts/gene_ga_data.py .
ln -s ../subscripts/get_best_result.sh .
ln -s ../subscripts/ipaexg.ttf .
ln -s ../subscripts/plot_ga.py .
ln -s ../subscripts/recalcu_confmat_cv_multi.py .
ln -s ../subscripts/run_ga.py .
ln -s ../subscripts/utils.py .

cd ../tools_for_shift_2
ln -s ../subscripts/evaluate.py .
ln -s ../subscripts/ga.py .
ln -s ../subscripts/gene_ga_data.py .
ln -s ../subscripts/get_best_result.sh .
ln -s ../subscripts/ipaexg.ttf .
ln -s ../subscripts/output_thresh_confmat.py .
ln -s ../subscripts/plot_confmat.py .
ln -s ../subscripts/plot_ga.py .
ln -s ../subscripts/plot_score_dist.py .
ln -s ../subscripts/run_ga.py .
ln -s ../subscripts/utils.py .

cd ../tools_for_shift_multi/
ln -s ../subscripts/evaluate.py .
ln -s ../subscripts/ga.py .
ln -s ../subscripts/gene_ga_data.py .
ln -s ../subscripts/get_best_result.sh .
ln -s ../subscripts/ipaexg.ttf .
ln -s ../subscripts/plot_ga.py .
ln -s ../subscripts/run_ga.py .
ln -s ../subscripts/utils.py .

exit 0
