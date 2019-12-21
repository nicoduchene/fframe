FFrame
======

FFrame was a quick comparison between two algorithms.
Once implemented, one was extremely inefficient, and the second much faster!
The goal was to discretize a function's domain and image
given some allowed values.

1. Look up table *like* method. 
    It comes down to populating a list of allowed values, 
    iterating over the discretized domain's image values, 
    and finding the nearest allowed value.
    This method would be more interesting if the values need
    to be used many times, instead of just once to discretize the 
    image values. There is a big problem with my implementation,
    due to some occasional mismatch between domain values and
    image values, depending on the input function. 
    I found these issues after implementing the much simpler method, 
    and so will not spend any time looking into the underlying error.
    It is most likely an indexing error. 
2. Iterator method.
    This is much simpler. It is a one-liner.
    It not only gets rid of a bunch of boiler-plate code,
    but it runs much faster. More on this in the comparison 
    of the two methods. Furthermore, there are no mismatch issues 
    to deal with, unless the function isn't defined in one of the 
    domain points.

Comparison of performance
-------------------------
TODO

