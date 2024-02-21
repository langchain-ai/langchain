#ifndef _util_h
#define _util_h

#define DBGS(__x) printf("%s\n", __x)
#define DBGI(__x) printf("%s: %i\n", #__x, __x)
#define DBGI2(__x, __y) printf("%s, %s: %i, %i\n", #__x, #__y, __x, __y)
#define DBGI3(__x, __y, __z) printf("%s, %s, %s: %i, %i, %i\n", #__x, #__y, #__z, __x, __y, __z)
#define DBGF(__x) printf("%s: %f\n", #__x, __x)
#define DBGF2(__x, __y) printf("%s, %s: %f, %f\n", #__x, #__y, __x, __y)
#define DBGF3(__x, __y, __z) printf("%s, %s, %s: %f, %f, %f\n", #__x, #__y, #__z, __x, __y, __z)

#endif
