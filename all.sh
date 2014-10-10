#!/usr/bin/gnuplot
	set title "CUDA shared vs. unshared"
        set xlabel "Matrix dimension"
        set ylabel "Time (ms)"
	set yrange [0:0.5]
        set term png
        set output "cuda.png"
	plot "graphs.out" using 3:2 with linespoints title "cuda-shared", "graphs.out" using 3:1 with linespoints title "cuda-unshared"
