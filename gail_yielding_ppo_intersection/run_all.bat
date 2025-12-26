@echo off
setlocal

cd /d "C:\Users\14487\python-book\yielding_imitation\Yielding_imitation_github\gail_yielding_ppo_intersection"

for /L %%W in (5,5,500) do (
    echo Running weight=%%W
    python run_yield.py --weight %%W
)

echo.
echo Done.
pause
