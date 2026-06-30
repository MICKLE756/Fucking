# 本简历使用中文（ctex/xeCJK），必须用 XeLaTeX 编译。
# 有了本文件，在 resume/ 目录直接运行 `latexmk` 即可（无需 -xelatex）。
$pdf_mode = 5;        # 5 = 使用 xelatex 生成 PDF
$xelatex = 'xelatex -interaction=nonstopmode -synctex=1 %O %S';
$bibtex_use = 0;      # 本简历无参考文献，禁用 bibtex，避免误触发
$clean_ext = 'synctex.gz aux log out fls fdb_latexmk';
