#!/usr/bin/perl

open(FO,"|sort |uniq >/tmp/t");
$i = 0;
while(<>) {
    next if /^pp/;

    ($x,$y,$z) = split(/\s+/);

    $x0 = int((2*$x+$z)*250+5000);
    $y0 = int((-$y-$z)*300+5000);


    print FO "$x0 $y0";
    print FO (($i == 2) ? "\n" : " ");
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
    chomp;
    print "2 3 0 1 0 7 100 0 10 0.000 0 0 -1 0 0 4\n";
    ($x0,$y0) = split(/\s+/);
    print "\t$_ $x0 $y0\n";
}
close(FIN);

