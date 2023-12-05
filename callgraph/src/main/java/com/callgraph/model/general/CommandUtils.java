package com.callgraph.model.general;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;

public class CommandUtils {
    // TODO This needs to understand the command and change it according to the OS type
    public static void runCommand(String locationPath, ArrayList<String> commands, boolean isWindows) {
        ProcessBuilder processBuilder = new ProcessBuilder();
        File location = new File(locationPath);
        processBuilder.directory(location);

        for (String command: commands) {
            if (isWindows) {
                processBuilder.command("cmd.exe", "/c", command);
            } else {
                processBuilder.command("sh", "-c", command);
            }

            try {
                Process p = processBuilder.start();
                p.waitFor();
            } catch (IOException | InterruptedException e) {
                System.out.println(e.getMessage());
            }
        }
    }

    // TODO This needs to understand the command and change it according to the OS type
    public static void runCommand(String locationPath, String command, boolean isWindows) {
        ProcessBuilder processBuilder = new ProcessBuilder();
        File location = new File(locationPath);
        processBuilder.directory(location);

        if (isWindows) {
            processBuilder.command("cmd.exe", "/c", command);
        } else {
            processBuilder.command("sh", "-c", command);
        }

        try {
            Process p = processBuilder.start();
            p.waitFor();
        } catch (IOException | InterruptedException e) {
            System.out.println(e.getMessage());
        }
    }
}
