#!/bin/bash
# Runs all tests
# Win32 path hacks
export CWD=$(pwd)
export PYMAJOR="$(python -c "import sys; print(sys.version_info[0])")"

# <CORRECT_PYTHON>
# GET CORRECT PYTHON ON ALL PLATFORMS
export SYSNAME="$(expr substr $(uname -s) 1 10)"
if [ "$SYSNAME" = "MINGW32_NT" ]; then
    export PYEXE=python
else
    if [ "$PYMAJOR" = "3" ]; then
        # virtual env?
        export PYEXE=python
    else
        export PYEXE=python2.7
    fi
fi
# </CORRECT_PYTHON>

PRINT_DELIMETER()
{
    printf "\n#\n#\n#>>>>>>>>>>> next_test\n\n"
}

export TEST_ARGV="--quiet --noshow $@"



# Default tests to run
set_test_flags()
{
    export DEFAULT=$1
    export DOC_TEST=$DEFAULT
}
set_test_flags OFF
export DOC_TEST=ON

# Parse for bash commandline args
for i in "$@"
do
case $i in --testall)
    set_test_flags ON
    ;;
esac
case $i in --notestdoc)
    export DOC_TEST=OFF
    ;;
esac
case $i in --testdoc)
    export DOC_TEST=ON
    ;;
esac
done

BEGIN_TESTS()
{
cat <<EOF
  ______ _     _ __   _      _______ _______ _______ _______ _______
 |_____/ |     | | \  |         |    |______ |______    |    |______
 |    \_ |_____| |  \_|         |    |______ ______|    |    ______|
                                                                    

EOF
    echo "BEGIN: TEST_ARGV=$TEST_ARGV"
    PRINT_DELIMETER
    num_passed=0
    num_ran=0
    export FAILED_TESTS=''
}

RUN_TEST()
{
    echo "RUN_TEST: $@"
    export TEST="$PYEXE $@ $TEST_ARGV"
    $TEST
    export RETURN_CODE=$?
    echo "RETURN_CODE=$RETURN_CODE"
    PRINT_DELIMETER
    num_ran=$(($num_ran + 1))
    if [ "$RETURN_CODE" == "0" ] ; then
        num_passed=$(($num_passed + 1))
    fi
    if [ "$RETURN_CODE" != "0" ] ; then
        export FAILED_TESTS="$FAILED_TESTS\n$TEST"
    fi
}

END_TESTS()
{
    echo "RUN_TESTS: DONE"
    if [ "$FAILED_TESTS" != "" ] ; then
        echo "-----"
        printf "Failed Tests:"
        printf "$FAILED_TESTS\n"
        printf "$FAILED_TESTS\n" >> failed.txt
        echo "-----"
    fi
    echo "$num_passed / $num_ran tests passed"
}

#---------------------------------------------
# START TESTS
BEGIN_TESTS

# Quick Tests (always run)

#---------------------------------------------
#DOC TESTS
if [ "$DOC_TEST" = "ON" ] ; then
cat <<EOF
    ___  ____ ____    ___ ____ ____ ___ ____ 
    |  \ |  | |        |  |___ [__   |  [__  
    |__/ |__| |___     |  |___ ___]  |  ___]
EOF
    RUN_TEST utool/util_class.py --test-make_class_method_decorator:0
    RUN_TEST utool/util_progress.py --test-ProgressIter:0
    RUN_TEST utool/util_progress.py --test-test_progress:0
    RUN_TEST utool/util_cache.py --test-cached_func:0
    RUN_TEST utool/util_list.py --test-depth_profile:0
    RUN_TEST utool/util_list.py --test-find_nonconsec_indicies:0
    RUN_TEST utool/util_list.py --test-ifilter_items:0
    RUN_TEST utool/util_list.py --test-ifilterfalse_items:0
    RUN_TEST utool/util_list.py --test-invertable_flatten2:0
    RUN_TEST utool/util_list.py --test-list_depth:0
    RUN_TEST utool/util_list.py --test-setdiff_ordered:0
    RUN_TEST utool/util_list.py --test-sortedby:0
    RUN_TEST utool/util_list.py --test-sortedby2:0
    RUN_TEST utool/util_list.py --test-unflatten2:0
    RUN_TEST utool/util_dict.py --test-all_dict_combinations_lbls:0
    RUN_TEST utool/util_dict.py --test-invert_dict:0
    RUN_TEST utool/util_time.py --test-Timer:0
    RUN_TEST utool/util_time.py --test-get_timedelta_str:0
    RUN_TEST utool/util_time.py --test-get_timestamp:0
    RUN_TEST utool/util_csv.py --test-make_csv_table:0
    RUN_TEST utool/util_iter.py --test-ichunks:0
    RUN_TEST utool/util_iter.py --test-ifilter_items:0
    RUN_TEST utool/util_iter.py --test-ifilterfalse_items:0
    RUN_TEST utool/util_iter.py --test-interleave:0
    RUN_TEST utool/util_parallel.py --test-ProgressIter:0
    RUN_TEST utool/util_str.py --test-align:0
    RUN_TEST utool/util_str.py --test-horiz_string:0
    RUN_TEST utool/util_str.py --test-seconds_str:0
    RUN_TEST utool/util_numpy.py --test-intersect2d:0
    RUN_TEST utool/util_numpy.py --test-sample_domain:0
    RUN_TEST utool/util_dbg.py --test-execstr_dict:0
    RUN_TEST utool/util_dbg.py --test-horiz_string:0
    RUN_TEST utool/util_grabdata.py --test-grab_file_url:0
    RUN_TEST utool/util_path.py --test-get_modname_from_modpath:0
    RUN_TEST utool/util_path.py --test-get_module_subdir_list:0
    RUN_TEST utool/util_path.py --test-get_relative_modpath:0
    RUN_TEST utool/util_path.py --test-is_module_dir:0
    RUN_TEST utool/util_path.py --test-path_ndir_split:0
    RUN_TEST utool/util_regex.py --test-named_field_regex:0
    RUN_TEST utool/util_regex.py --test-regex_replace:0
    RUN_TEST utool/util_tests.py --test-bubbletext:0
    RUN_TEST utool/util_tests.py --test-get_doctest_examples:0
    RUN_TEST utool/util_tests.py --test-tryimport:0
    RUN_TEST utool/util_alg.py --test-cartesian:0
    RUN_TEST utool/util_alg.py --test-group_items:0
fi

#---------------------------------------------
# END TESTING
END_TESTS