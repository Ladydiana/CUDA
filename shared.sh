#!/usr/bin/gnuplot

        set title "SHARED"
        set xlabel "Matrix dimension"
        set ylabel "Time (ms)"
	set yrange [-0.5:50]
        set term png
        set output "shared.png"
        plot "graphs.out" using 3:4 with linespoints title "gold", "graphs.out" using 3:2 with linespoints title "cuda-shared"

