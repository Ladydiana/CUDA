#!/usr/bin/gnuplot

	set title "UNSHARED"
        set xlabel "Matrix dimension"
        set ylabel "Time (ms)"
	set yrange [-0.5:50]
        set term png
        set output "unshared.png"
	plot "graphs.out" using 3:4 with linespoints title "gold", "graphs.out" using 3:1 with linespoints title "cuda-unshared"
