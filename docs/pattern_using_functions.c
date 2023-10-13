//         1 
//       1 2 1 
//     1 2 3 2 1         
//   1 2 3 4 3 2 1  

#include <stdio.h>

void primeNum(int n)
{
    for (int i = 1; i <= n; i++)
    {
        printf("%d ", i);
    }
}
void primeNumReverse(int n)
{
    for (int i = n; i >= 1; i--)
    {
        printf("%d ", i);
    }
}
void spaces(int n)
{
    for (int j = n - 1; j >= 0; j--)
    {
        printf("  ");
    }
}
int main()
{
    int n = 5;
    spaces(4);
    primeNum(1);
    primeNumReverse(0);
    printf("\n");
    spaces(3);
    primeNum(2);
    primeNumReverse(1);
    printf("\n");
    spaces(2);
    primeNum(3);
    primeNumReverse(2);
    printf("\n");
    spaces(1);
    primeNum(4);
    primeNumReverse(3);
    printf("\n");
}

