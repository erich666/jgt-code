#!/usr/bin/perl

open(FO,"|sort |uniq >/tmp/t");
$i = 0;
while(<>) {
    next if /^pp/;


    ($x,$y,$z) = split(/\s+/);
    $x[$i] = int((2*$x+$z)*250+5000);
    $y[$i] = int((-$y-$z)*300+5000);

    if ($i==2) {
	for ($j = 0; $j<3; $j++) {
	    $k = ($j+1)%3;
	    if ($x[$j]<$x[$k] || ($x[$j]==$x[$k] && $y[$j]<$y[$k])) {
		print FO "$x[$j] $y[$j] $x[$k] $y[$k]\n";
	    }
	    else {
		print FO "$x[$k] $y[$k] $x[$j] $y[$j]\n";
	    }
	}
    }
    $i = ($i+1)%3;
}
close(FO);

print << 'ENDEND';
#FIG 3.2
Landscape
Center
Inches
Letter  
100.00
Single
-2
1200 2
ENDEND

open(FIN,"/tmp/t");
while(<FIN>) {
    print "2 1 0 1 0 7 100 0 -1 0.000 0 0 -1 0 0 2\n";
    print "\t$_";
}
close(FIN);

