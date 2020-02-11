##### CODE DOWNLOADED FROM https://people.sc.fsu.edu/~jburkardt/py_src/asa103/digamma.py
##### RESPECT ITS LICENCE

#! /usr/bin/env python
#
def digamma ( x ):

#*****************************************************************************80
#
## DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    20 March 2016
#
#  Author:
#
#    Original FORTRAN77 version by Jose Bernardo.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Jose Bernardo,
#    Algorithm AS 103:
#    Psi ( Digamma ) Function,
#    Applied Statistics,
#    Volume 25, Number 3, 1976, pages 315-317.
#
#  Parameters:
#
#    Input, real X, the argument of the digamma function.
#    0 < X.
#
#    Output, real DIGAMMA, the value of the digamma function at X.
#
#    Output, integer IFAULT, error flag.
#    0, no error.
#    1, X <= 0.
#
  import numpy as np
#
#  Check the input.
#
  if ( x <= 0.0 ):
    value = 0.0
    ifault = 1
    return value, ifault
#
#  Initialize.
#
  ifault = 0
  value = 0.0
#
#  Use approximation for small argument.
#
  if ( x <= 0.000001 ):
    euler_mascheroni = 0.57721566490153286060
    value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
    return value, ifault
#
#  Reduce to DIGAMA(X + N).
#
  while ( x < 8.5 ):
    value = value - 1.0 / x
    x = x + 1.0
#
#  Use Stirling's (actually de Moivre's) expansion.
#
  r = 1.0 / x
  value = value + np.log ( x ) - 0.5 * r
  r = r * r
  value = value \
    - r * ( 1.0 / 12.0 \
    - r * ( 1.0 / 120.0 \
    - r * ( 1.0 / 252.0 \
    - r * ( 1.0 / 240.0 \
    - r * ( 1.0 / 132.0 ) ) ) ) )

  return value, ifault

def digamma_test ( ):

#*****************************************************************************80
#
## DIGAMMA_TEST tests DIGAMMA.
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    17 March 2016
#
#  Author:
#
#    John Burkardt
#
  import platform
  from psi_values import psi_values

  print ( '' )
  print ( 'DIGAMMA_TEST:' )
  print ( '  Python version: %s' % ( platform.python_version ( ) ) )
  print ( '  DIGAMMA computes the Digamma or Psi function.' )
  print ( '  Compare the result to tabulated values.' )
  print ( '' )
  print ( '          X         FX                        FX2' )
  print ( '                    (Tabulated)               (DIGAMMA)               DIFF' )
  print ( '' )

  n_data = 0

  while ( True ):

    n_data, x, fx = psi_values ( n_data )

    if ( n_data == 0 ):
      break

    fx2, ifault = digamma ( x )

    print ( '  %12.4f  %24.16e  %24.16e  %10.4e' % ( x, fx, fx2, abs ( fx - fx2 ) ) )
#
#  Terminate.
#
  print ( '' )
  print ( 'DIGAMMA_TEST:' )
  print ( '  Normal end of execution.' )
  return

if ( __name__ == '__main__' ):
  from timestamp import timestamp
  timestamp ( )
  digamma_test ( )
  timestamp ( )
