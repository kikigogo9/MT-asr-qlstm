:loop
py gridsearch.py %1 %2 %3 %4 %5
echo Script crashed. Restarting...
timeout /t 1 >nul
goto loop