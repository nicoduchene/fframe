FFrame
======

FFrame was a quick comparison between two algorithms.
One was extremely inefficient, and the second much faster!
The goal was to discretize a function's domain and image
given some allowed values.

Dependencies:
    - numpy
    - scipy
    - matplotlib

Questions:
    - What is the operational overhead of implementing an iterative 
    function dicretizer instead of a LUT method?
        - Depends on the division algorithm used in python.
            - One could round to nearest integer before finding nearest allowed value.
                - This would incur floating point errors 
            - One could also spread seeds in the image ensemble and define neighborhoods 
            around seeds where lut values are stored.
            - Apparently python isn't very good at fast division.

    - What is the most energetically efficient implementation?
        - How does it depend on the hardware being used?
        - How does it depend on the language being used?
