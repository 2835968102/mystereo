find_program(PYTHON3_EXECUTABLE NAMES python3)
if(PYTHON3_EXECUTABLE)
  add_test(
    NAME run_stereo_calib_smoke
    COMMAND ${PYTHON3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/run_stereo_calib_smoke.py
            --binary $<TARGET_FILE:run_stereo_calib>
  )
else()
  message(WARNING "python3 not found; run_stereo_calib_smoke test will not be registered.")
endif()
