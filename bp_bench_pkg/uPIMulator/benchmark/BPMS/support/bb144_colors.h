#ifndef BPMS_BB144_COLORS_H
#define BPMS_BB144_COLORS_H

#include <stdint.h>

#define BB_NUM_COLORS 3
#define BB_CHECKS_PER_COLOR 24

/* Every variable occurs in exactly one check per color. */
static const uint16_t bb_check_colors[BB_NUM_COLORS][BB_CHECKS_PER_COLOR] = {
    {1,4,6,9,14,17,19,22,24,27,32,35,37,40,42,45,50,53,55,58,60,63,68,71},
    {0,3,8,11,13,16,18,21,26,29,31,34,36,39,44,47,49,52,54,57,62,65,67,70},
    {2,5,7,10,12,15,20,23,25,28,30,33,38,41,43,46,48,51,56,59,61,64,66,69},
};

/* Training-selected deterministic color traversal order. */
static const uint16_t bb_color_order[BB_NUM_COLORS] = {1, 2, 0};

#endif
